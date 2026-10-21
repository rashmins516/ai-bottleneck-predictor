"""Phase 2/3 tests: causality of features, future-only labels, purged splits."""
import numpy as np
import pandas as pd
import pytest

from iobn.features import (build_device_features, feature_columns, build_features_for_runs,
                           process_rates_at_ticks)
from iobn.labeling import (BottleneckCriteria, LabelConfig, make_labels, purged_split,
                           purged_walk_forward)
from iobn.model import TrainConfig, assemble

from synth import synth_run


def test_features_are_strictly_causal():
    df, _ = synth_run(seconds=200, seed=1)
    F0 = build_device_features(df)
    k = 400
    df2 = df.copy()
    rng = np.random.default_rng(9)
    for c in ("d_rd_ios", "d_wr_ticks", "d_time_in_queue", "mem_dirty_kb", "d_psi_some_us"):
        df2.loc[df2.index > k, c] = rng.uniform(0, 1e6, (df2.index > k).sum())
    F1 = build_device_features(df2)
    cols = feature_columns(F0)
    pd.testing.assert_frame_equal(F0.loc[:k, cols], F1.loc[:k, cols])


def test_multiscale_volatility_acceleration_and_process_features():
    df, _ = synth_run(seconds=120, seed=11)
    F = build_device_features(df)
    assert {"await_volatility_ms_1s", "await_volatility_ms_5s",
            "throughput_volatility_mbps_30s", "latency_accel_ms_s2",
            "queue_accel_s2", "throughput_accel_mbps_s2"} <= set(F)

    ticks = pd.DataFrame({"ts": np.arange(0.0, 20.0, 0.25), "valid": 1})
    proc = pd.DataFrame([{"ts": 10.0, "dt_s": 1.0, "d_rchar": 1000, "d_wchar": 3000,
                          "d_read_bytes": 100, "d_write_bytes": 250,
                          "d_cancelled_write_bytes": 50}])
    rates = process_rates_at_ticks(ticks, proc, max_age_s=2.0)
    assert rates.loc[ticks["ts"] == 10.0, "proc_io_bps"].iloc[0] == 300
    assert rates.loc[ticks["ts"] == 12.0, "proc_write_bps"].iloc[0] == 200
    assert rates.loc[ticks["ts"] == 10.0, "proc_submit_bps"].iloc[0] == 4000
    assert rates.loc[ticks["ts"] == 12.25, "proc_io_bps"].isna().all()


def test_process_rate_features_do_not_use_future_scans():
    df, _ = synth_run(seconds=120, seed=12)
    df["run_id"] = "r1"
    t0 = float(df["ts"].iloc[0])
    proc = pd.DataFrame([
        {"run_id": "r1", "ts": t0 + 5.0, "dt_s": 1.0, "d_rchar": 1000, "d_wchar": 3000,
         "d_read_bytes": 100, "d_write_bytes": 250, "d_cancelled_write_bytes": 50},
        {"run_id": "r1", "ts": t0 + 10.0, "dt_s": 1.0, "d_rchar": 1000, "d_wchar": 9000,
         "d_read_bytes": 100, "d_write_bytes": 250, "d_cancelled_write_bytes": 50},
    ])
    F0 = build_features_for_runs(df, "sdx", proc_all=proc)
    proc.loc[proc["ts"] == t0 + 10.0, "d_wchar"] = 9_000_000
    F1 = build_features_for_runs(df, "sdx", proc_all=proc)
    cols = feature_columns(F0)
    pd.testing.assert_frame_equal(F0.loc[df["ts"] <= t0 + 9.0, cols],
                                  F1.loc[df["ts"] <= t0 + 9.0, cols])


def test_label_comes_only_from_future_window():
    n, dt = 200, 0.25
    ts = np.arange(n) * dt
    F = pd.DataFrame({"await_1s": 0.3, "iops_1s": 100.0, "psi_some_1s": 0.0, "valid": 1}, index=range(n))
    u = 100
    F.loc[u:u + 3, "await_1s"] = 50.0                  # state pulse (debounce needs >= 0.5 s)
    crit = BottleneckCriteria(baseline_await_ms=0.3)
    L = make_labels(F, ts, crit, LabelConfig(1.5, 3.0))
    s_times = ts[(L["state_now"] == 1).to_numpy()]
    assert len(s_times) > 0
    # y(t) = 1  <=>  some pressured tick u with t + 1.5 < u <= t + 3.0
    expected = np.array([((s_times > t + 1.5) & (s_times <= t + 3.0)).any() for t in ts])
    ok = L["label_ok"].to_numpy().astype(bool)
    assert np.array_equal(L["y"].to_numpy()[ok].astype(bool), expected[ok])
    # Rows labelled positive all lie strictly BEFORE the pressured period:
    # the model is never asked to recognise a bottleneck that already exists.
    assert ts[(L["y"] == 1).to_numpy()].max() < s_times.min()


def test_overlapping_label_window_is_rejected():
    F = pd.DataFrame({"await_1s": [0.3] * 20, "iops_1s": [100.0] * 20, "valid": 1})
    with pytest.raises(ValueError, match="leakage"):
        make_labels(F, np.arange(20) * 0.25, BottleneckCriteria(baseline_await_ms=0.3),
                    LabelConfig(h_start_s=1.0, h_end_s=3.0))


def test_incomplete_future_is_nan_not_zero():
    df, _ = synth_run(seconds=120, seed=2)
    F = build_device_features(df)
    F["valid"] = 1
    L = make_labels(F, df["ts"].to_numpy(), BottleneckCriteria(baseline_await_ms=0.3))
    tail = df["ts"] > df["ts"].iloc[-1] - 3.0
    assert L.loc[tail, "y"].isna().all()


def test_purged_split_has_no_shared_raw_data():
    ts = np.arange(0, 600, 0.25)
    D = pd.DataFrame({"ts": ts, "run_id": "r1"})
    s = purged_split(D, 0.3, h_end_s=3.0, max_lookback_s=30.0)
    train_last_label_data = ts[s.train_idx].max() + 3.0
    test_first_feature_data = ts[s.test_idx].min() - 30.0
    assert train_last_label_data < test_first_feature_data + 1e-9
    for tr, te in purged_walk_forward(ts, 3, 3.0, 30.0):
        assert ts[tr].max() + 3.0 < ts[te].min() - 30.0 + 1e-9


def test_baselines_fit_on_train_only():
    df, _ = synth_run(seconds=600, seed=3)
    df["run_id"] = "r1"
    cfg = TrainConfig(device="sdx")
    _, split, crit_a, norm_a = assemble(df, cfg)
    df2 = df.copy()
    test_ts = df2["ts"] >= df2["ts"].iloc[split.test_idx.min()] - 30
    df2.loc[test_ts, "d_rd_ticks"] *= 100                # wreck the test period
    _, _, crit_b, norm_b = assemble(df2, cfg)
    assert crit_a.baseline_await_ms == pytest.approx(crit_b.baseline_await_ms)
    assert norm_a.base_await_ms == pytest.approx(norm_b.base_await_ms)
