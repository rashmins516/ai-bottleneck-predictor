"""End-to-end (synthetic) training and Phase 4 attribution tests."""
import json
import sys
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pandas as pd

from iobn.attribution import (AttributionConfig, attribute_predicted_bottleneck_json,
                              rank_likely_contributors)
from iobn.model import (TrainConfig, _threshold_search, alarm_policy_by_group, train)

from synth import synth_run


def test_threshold_search_enforces_precision_floor_after_debounce():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.95, 0.92, 0.90, 0.89, 0.20, 0.10])
    groups = np.array(["run1"] * len(y))
    chosen = _threshold_search(y, p, min_precision=0.80, k=2, groups=groups)
    alarms = alarm_policy_by_group(p, groups, chosen["threshold"], 2)
    assert chosen["precision_floor_met"]
    assert chosen["precision"] >= 0.80
    assert np.array_equal(alarms, [False, True, False, False, False, False])


def test_alarm_debounce_resets_between_runs():
    alarms = alarm_policy_by_group(np.array([0.99, 0.99]), np.array(["a", "b"]), 0.5, 2)
    assert not alarms.any()


def test_model_forecasts_onsets_that_persistence_cannot(tmp_path):
    runs = []
    for seed in range(4):
        df, _ = synth_run(seconds=900, seed=seed, n_episodes=30)
        d = tmp_path / f"run{seed}"
        d.mkdir()
        df.to_csv(d / "device.csv", index=False)
        runs.append(str(d))
    rep = train(runs, TrainConfig(device="sdx"), importance=False)
    m = rep["metrics"]
    onset_model = m["onset"]["model"]["recall"]
    onset_persist = m["onset"]["persistence"]["recall"]
    assert onset_persist == 0.0                     # by construction, persistence can't forecast onsets
    assert onset_model > 0.5                        # predictable (ramped) onsets are caught
    assert m["episodes_model"]["warned_ahead_frac"] > 0.4
    assert m["episodes_model"]["lead_time_vs_raw_onset_s"]["median"] >= 1.5


def _proc_rows(t_ticks, pid, start, comm, phys_series, rchar=0, kthread=0, blk=0):
    return [{"ts": t, "pid": pid, "start_ticks": start, "comm": comm, "ppid": 1, "kthread": kthread,
             "born_in_interval": 0, "reaped_child_adjusted": 0, "d_blkio_ticks": blk,
             "d_rchar": rchar, "d_wchar": p, "d_syscr": 10, "d_syscw": max(1, int(p // 4096)),
             "d_read_bytes": 0, "d_write_bytes": p, "d_cancelled_write_bytes": 0}
            for t, p in zip(t_ticks, phys_series)]


def test_attribution_ranks_coupled_writer_over_cache_reader_and_kernel_threads():
    t = np.arange(1, 21) * 0.5
    culprit = np.where((t > 4) & (t < 8), 40e6, 1e5)        # bursts that line up with latency
    steady = np.full(len(t), 2e6)                            # constant, uncoupled
    rows = (_proc_rows(t, 100, 1, "culprit", culprit)
            + _proc_rows(t, 200, 2, "steady", steady)
            + _proc_rows(t, 300, 3, "decoy", np.zeros(len(t)), rchar=500e6)   # page-cache reads
            + _proc_rows(t, 7, 0, "jbd2/sda1-8", np.full(len(t), 5e6), kthread=1)
            + _proc_rows(t, 400, 4, "victim", np.full(len(t), 1e3), blk=40))
    P = pd.DataFrame(rows)
    dts = np.arange(1, 81) * 0.125
    lat = np.where((dts > 4.25) & (dts < 8.5), 30.0, 0.3)
    ios = np.full(len(dts), 100.0)
    D = pd.DataFrame({"ts": dts, "device": "sdx", "valid": 1, "d_rd_ios": 0.0, "d_wr_ios": ios,
                      "d_rd_ticks": 0.0, "d_wr_ticks": ios * lat,
                      "d_rd_sectors": 0.0, "d_wr_sectors": np.full(len(dts), 2e6 / 512)})
    rep = rank_likely_contributors(P, D, "sdx", 10.0, AttributionConfig(lookback_s=10))
    names = [c.comm for c in rep.contributors]
    assert names[0] == "culprit"
    assert "decoy" not in names[:2]
    assert "jbd2/sda1-8" not in names and any(n.startswith("jbd2") for _, n, _ in rep.kernel_proxies)
    assert rep.likely_victims and rep.likely_victims[0][1] == "victim"
    assert rep.contributors[0].coupling_rho > 0.5
    log = rep.to_log(0.9)
    assert "LIKELY CONTRIBUTORS" in log and "not proven causes" in log


def test_prediction_gated_attribution_returns_json_and_process_shares(monkeypatch):
    t = np.arange(1, 21) * 0.5
    writes = np.full(len(t), 4000.0)
    proc = pd.DataFrame(_proc_rows(t, 100, 1, "writer", writes))
    dts = np.arange(1, 81) * 0.125
    dev = pd.DataFrame({"ts": dts, "device": "sdx", "valid": 1, "d_rd_ios": 0.0,
                        "d_wr_ios": 10.0, "d_rd_ticks": 0.0, "d_wr_ticks": 30.0,
                        "d_rd_sectors": 0.0, "d_wr_sectors": 8.0})
    key_type = namedtuple("ProcKey", ["pid", "start_ticks"])
    monkeypatch.setitem(sys.modules, "iobn.telemetry", SimpleNamespace(
        scan_processes=lambda: ({key_type(100, 1): object()}, SimpleNamespace(
            io_denied=0, vanished=0, malformed=0)),
        CLK_TCK=100,
    ))
    monkeypatch.setattr("iobn.attribution.os.readlink", lambda _: "/usr/bin/writer")

    assert attribute_predicted_bottleneck_json(
        proc, dev, "sdx", 10.0, 0.4, 0.5, AttributionConfig(lookback_s=10)
    ) is None
    payload = attribute_predicted_bottleneck_json(
        proc, dev, "sdx", 10.0, 0.9, 0.5, AttributionConfig(lookback_s=10)
    )
    assert payload is not None
    report = json.loads(payload)
    writer = report["contributors"][0]
    assert report["prediction"]["triggered"]
    assert writer["pid"] == 100
    assert writer["command_path"] == "/usr/bin/writer"
    assert writer["write_bytes"] > 0
    assert writer["write_share_pct"] == 100.0
    assert writer["block_wait_seconds"] is not None
    assert "victim evidence" in report["block_wait_semantics"]
