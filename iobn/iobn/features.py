"""Phase 2 — Causal device-pressure features (no queue-overflow assumption).

Why not "queue overflow"?
    On blk-mq NVMe/SSD stacks, requests are spread across many hardware
    queues with deep tags; the queue essentially never "overflows" before the
    application suffers. What actually happens near saturation:
      * per-request latency (await) rises,
      * throughput stops scaling with offered concurrency (Little's law:
        aqu = IOPS x await, so extra in-flight requests turn into extra
        latency instead of extra IOPS),
      * tasks start stalling (PSI io.some),
      * for buffered writes, Dirty page volume builds up before writeback
        hits the device.
    Every feature below measures one of those, over sliding windows that end
    at time t (strictly causal).

Correct aggregation:
    Latency over a window is sum(d_ticks)/sum(d_ios), i.e. I/O-weighted, not
    the mean of per-tick awaits. diskstats ticks have 1 ms resolution, so
    per-tick awaits on fast or virtual disks are quantised to 0/1/2 ms;
    window aggregation is what makes them usable at all.

Causality guarantee:
    pandas time-based rolling windows are right-aligned and closed on the
    right: the window for row t covers (t - w, t]. No center=True, no
    negative shifts, no bfill. tests/test_leakage.py perturbs future rows and
    asserts features at t do not change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SECTOR = 512
META_COLS = {"ts", "wall", "dt_s", "device", "valid", "gap", "suspend", "run_id"}


@dataclass
class FeatureConfig:
    windows_s: Sequence[float] = (1.0, 5.0, 30.0)
    trend_window_s: float = 5.0
    min_ios_for_latency: int = 8      # fewer completions -> latency NaN (too noisy)
    process_max_age_s: float = 2.0

    @property
    def max_lookback_s(self) -> float:
        # Second derivatives regress a trend which already uses one rolling window.
        return max(max(self.windows_s), 2.0 * self.trend_window_s + min(self.windows_s),
                   self.process_max_age_s)


def _wstr(w: float) -> str:
    return f"{int(w)}s" if float(w).is_integer() else f"{w}s"


RAW_INPUT_COLS = ("ts", "valid", "dt_s", "d_rd_ios", "d_wr_ios", "d_rd_ticks", "d_wr_ticks",
                  "d_rd_sectors", "d_wr_sectors", "d_rd_merges", "d_wr_merges", "d_io_ticks",
                  "d_time_in_queue", "d_fl_ios", "d_psi_some_us", "d_psi_full_us", "iowait_frac",
                  "steal_frac", "mem_dirty_kb", "mem_writeback_kb", "in_flight",
                  "proc_io_bps", "proc_write_bps", "proc_max_io_bps",
                  "proc_submit_bps", "proc_write_submit_bps")
PROCESS_RATE_COLS = RAW_INPUT_COLS[-5:]


class _Windows:
    """Causal time-window sums via prefix sums.

    For row i and window w the window is (ts[i] - w, ts[i]] — identical to
    pandas' right-closed offset windows, but O(n) numpy instead of dozens of
    rolling() calls. The same code path is used for training and for live
    inference, so there is no train/serve skew, and live scoring costs ~1 ms.
    """

    def __init__(self, ts: np.ndarray):
        self.ts = ts
        self._lo: Dict[float, np.ndarray] = {}

    def lo(self, w: float) -> np.ndarray:
        if w not in self._lo:
            self._lo[w] = np.searchsorted(self.ts, self.ts - w, side="right")
        return self._lo[w]

    def sums(self, M: np.ndarray, w: float, prefix=None) -> np.ndarray:
        """Window sums of every column of M (n x k) at once, skipping NaN; NaN
        where a column has no finite value in the window. Pass a precomputed
        `prefix(M)` to reuse the cumulative sums across several windows."""
        cs, cn = prefix if prefix is not None else self.prefix(M)
        lo = self.lo(w)
        idx = np.arange(1, len(self.ts) + 1)
        out = cs[idx] - cs[lo]
        out[(cn[idx] - cn[lo]) == 0] = np.nan
        return out

    @staticmethod
    def prefix(M: np.ndarray):
        fin = np.isfinite(M)
        z = np.zeros((1, M.shape[1]))
        return (np.vstack([z, np.cumsum(np.where(fin, M, 0.0), axis=0)]),
                np.vstack([z, np.cumsum(fin, axis=0)]))

    def sum(self, x: np.ndarray, w: float) -> np.ndarray:
        return self.sums(x[:, None], w)[:, 0]

    def count(self, x: np.ndarray, w: float) -> np.ndarray:
        cn = np.concatenate([[0], np.cumsum(np.isfinite(x))])
        return (cn[np.arange(1, len(x) + 1)] - cn[self.lo(w)]).astype(float)

    def mean(self, x: np.ndarray, w: float) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum(x, w) / self.count(x, w)

    def slopes(self, pairs, w: float, min_n: int = 4) -> List[np.ndarray]:
        """Least-squares slope of y on x over the window for several (y, x)
        pairs at once (only rows where both are finite take part)."""
        cols = []
        for y, x in pairs:
            m = np.isfinite(y) & np.isfinite(x)
            xx, yy = np.where(m, x, np.nan), np.where(m, y, np.nan)
            cols += [np.where(m, 1.0, np.nan), xx, yy, xx * yy, xx * xx]
        S = self.sums(np.column_stack(cols), w)
        out = []
        for i in range(len(pairs)):
            n, Sx, Sy, Sxy, Sxx = (np.nan_to_num(S[:, 5 * i + j]) for j in range(5))
            with np.errstate(invalid="ignore", divide="ignore"):
                den = n * Sxx - Sx ** 2
                sl = (n * Sxy - Sx * Sy) / den
            sl[(n < min_n) | ~(np.abs(den) > 1e-9 * np.maximum(n ** 2, 1))] = np.nan
            out.append(sl)
        return out

    def slope(self, y: np.ndarray, x: np.ndarray, w: float, min_n: int = 4) -> np.ndarray:
        return self.slopes([(y, x)], w, min_n)[0]


def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


def aggregate_process_rates(proc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate physical and logical process I/O rates at each scan timestamp."""
    cols = ["ts", *PROCESS_RATE_COLS]
    required = {"ts", "dt_s", "d_read_bytes", "d_write_bytes", "d_cancelled_write_bytes",
                "d_rchar", "d_wchar"}
    if proc.empty or not required.issubset(proc.columns):
        return pd.DataFrame(columns=cols)

    p = proc[list(required)].copy()
    for name in p.columns:
        p[name] = pd.to_numeric(p[name], errors="coerce")
    p = p[np.isfinite(p["ts"]) & np.isfinite(p["dt_s"]) & (p["dt_s"] > 0)]
    if p.empty:
        return pd.DataFrame(columns=cols)

    p["proc_write_bps"] = (p["d_write_bytes"] - p["d_cancelled_write_bytes"]).clip(lower=0) / p["dt_s"]
    p["proc_io_bps"] = (p["d_read_bytes"].clip(lower=0) / p["dt_s"]) + p["proc_write_bps"]
    p["proc_max_io_bps"] = p["proc_io_bps"]
    p["proc_submit_bps"] = (p["d_rchar"].clip(lower=0) + p["d_wchar"].clip(lower=0)) / p["dt_s"]
    p["proc_write_submit_bps"] = p["d_wchar"].clip(lower=0) / p["dt_s"]
    return (p.groupby("ts", as_index=False)
            .agg(proc_io_bps=("proc_io_bps", "sum"),
                 proc_write_bps=("proc_write_bps", "sum"),
                 proc_max_io_bps=("proc_max_io_bps", "max"),
                 proc_submit_bps=("proc_submit_bps", "sum"),
                 proc_write_submit_bps=("proc_write_submit_bps", "sum"))[cols])


def process_rates_at_ticks(dev: pd.DataFrame, proc: pd.DataFrame,
                           max_age_s: float = 2.0) -> pd.DataFrame:
    """Causally align sparse process rates to device ticks, expiring stale scans."""
    result = pd.DataFrame(index=dev.index, columns=PROCESS_RATE_COLS, dtype=float)
    rates = aggregate_process_rates(proc)
    if rates.empty:
        return result

    left = pd.DataFrame({"_index": np.arange(len(dev)),
                         "_ts": pd.to_numeric(dev["ts"], errors="coerce").to_numpy()})
    left = left.sort_values("_ts")
    right = rates.rename(columns={"ts": "_process_ts"}).sort_values("_process_ts")
    aligned = pd.merge_asof(left, right, left_on="_ts", right_on="_process_ts",
                            direction="backward", tolerance=max_age_s)
    aligned = aligned.sort_values("_index")
    for name in PROCESS_RATE_COLS:
        result[name] = aligned[name].to_numpy()
    return result


def _rolling_std(W: _Windows, x: np.ndarray, window_s: float, min_n: int = 2) -> np.ndarray:
    count = W.count(x, window_s)
    sx = W.sum(x, window_s)
    sx2 = W.sum(x * x, window_s)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (sx2 - sx * sx / count) / (count - 1.0)
    out = np.sqrt(np.maximum(variance, 0.0))
    out[(count < min_n) | ~np.isfinite(variance)] = np.nan
    return out


def _rolling_corr(W: _Windows, x: np.ndarray, y: np.ndarray, window_s: float,
                  min_n: int = 4) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    x0, y0 = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    count = W.count(np.where(valid, 1.0, np.nan), window_s)
    sx, sy = W.sum(x0, window_s), W.sum(y0, window_s)
    sxx, syy = W.sum(x0 * x0, window_s), W.sum(y0 * y0, window_s)
    sxy = W.sum(x0 * y0, window_s)
    cov = sxy - sx * sy / count
    vx = sxx - sx * sx / count
    vy = syy - sy * sy / count
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cov / np.sqrt(vx * vy)
    out[(count < min_n) | (vx <= 0) | (vy <= 0)] = np.nan
    return np.clip(out, -1.0, 1.0)


def device_features_arrays(c: Dict[str, np.ndarray], cfg: FeatureConfig = FeatureConfig()
                           ) -> Dict[str, np.ndarray]:
    """Core feature computation on column arrays of ONE device of ONE run.

    Invalid ticks (counter reset, collector gap, suspend) contribute nothing to
    any window: their raw deltas are NaN, which window sums skip.
    """
    ts = np.asarray(c["ts"], float)
    n_rows = len(ts)
    if n_rows > 1 and np.any(np.diff(ts) < 0):
        raise ValueError("device rows must be sorted by ts")
    valid = np.asarray(c["valid"]).astype(bool)

    def col(name, default=np.nan):
        v = c.get(name)
        if v is None:
            return np.full(n_rows, default)
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            return v.copy()                    # fast path (live ring buffers)
        return np.array(pd.to_numeric(pd.Series(v), errors="coerce"), dtype=float)  # writable copy

    raw = {
        "dt_ms": col("dt_s") * 1000.0,
        "rd_ios": col("d_rd_ios"), "wr_ios": col("d_wr_ios"),
        "rd_ticks": col("d_rd_ticks"), "wr_ticks": col("d_wr_ticks"),
        "rd_sec": col("d_rd_sectors"), "wr_sec": col("d_wr_sectors"),
        "merges": col("d_rd_merges") + col("d_wr_merges"),
        "io_ticks": col("d_io_ticks"), "tiq": col("d_time_in_queue"),
        "fl_ios": col("d_fl_ios", 0.0),
        "psi_some": col("d_psi_some_us"), "psi_full": col("d_psi_full_us"),
        "iowait": col("iowait_frac"), "steal": col("steal_frac"),
        "proc_io_bps": col("proc_io_bps"), "proc_write_bps": col("proc_write_bps"),
        "proc_max_io_bps": col("proc_max_io_bps"),
        "proc_submit_bps": col("proc_submit_bps"),
        "proc_write_submit_bps": col("proc_write_submit_bps"),
    }
    for k in raw:
        raw[k][~valid] = np.nan

    W = _Windows(ts)
    f: Dict[str, np.ndarray] = {}
    med_dt = np.nanmedian(raw["dt_ms"]) / 1000.0 if np.isfinite(raw["dt_ms"]).any() else np.nan
    tick_ios = raw["rd_ios"] + raw["wr_ios"]
    tick_wait_ms = np.where(tick_ios > 0, _div(raw["rd_ticks"] + raw["wr_ticks"], tick_ios), np.nan)
    tick_throughput_mbps = _div((raw["rd_sec"] + raw["wr_sec"]) * SECTOR, raw["dt_ms"]) / 1e6
    keys = ("dt_ms", "rd_ios", "wr_ios", "rd_ticks", "wr_ticks", "rd_sec", "wr_sec", "merges",
            "io_ticks", "tiq", "fl_ios", "psi_some", "psi_full", "iowait", "steal")
    M = np.column_stack([raw[k] for k in keys])
    pre = W.prefix(M)                     # one cumulative sum, reused by every window
    cnt_all = pre[1]
    idx = np.arange(1, n_rows + 1)
    for w in cfg.windows_s:
        ws = _wstr(w)
        Sm = W.sums(M, w, pre)
        S = {k: Sm[:, i] for i, k in enumerate(keys)}
        lo = W.lo(w)
        cnt = {k: (cnt_all[idx, i] - cnt_all[lo, i]) for i, k in enumerate(keys)}
        secs = S["dt_ms"] / 1000.0
        n = S["rd_ios"] + S["wr_ios"]
        mi = cfg.min_ios_for_latency
        f[f"await_{ws}"] = np.where(n >= mi, _div(S["rd_ticks"] + S["wr_ticks"], n), np.nan)
        f[f"r_await_{ws}"] = np.where(S["rd_ios"] >= mi, _div(S["rd_ticks"], S["rd_ios"]), np.nan)
        f[f"w_await_{ws}"] = np.where(S["wr_ios"] >= mi, _div(S["wr_ticks"], S["wr_ios"]), np.nan)
        f[f"iops_{ws}"] = _div(n, secs)
        tot_sec = S["rd_sec"] + S["wr_sec"]
        f[f"bps_{ws}"] = _div(tot_sec * SECTOR, secs)
        f[f"write_byte_frac_{ws}"] = np.where(tot_sec > 0, _div(S["wr_sec"], tot_sec), np.nan)
        f[f"avg_req_kb_{ws}"] = np.where(n > 0, _div(tot_sec * SECTOR / 1024, n), np.nan)
        f[f"aqu_{ws}"] = _div(S["tiq"], S["dt_ms"])
        f[f"util_{ws}"] = _div(S["io_ticks"], S["dt_ms"])
        f[f"flush_rate_{ws}"] = _div(S["fl_ios"], secs)
        f[f"merge_ratio_{ws}"] = np.where(n > 0, _div(S["merges"], n + S["merges"]), np.nan)
        f[f"psi_some_{ws}"] = _div(S["psi_some"], S["dt_ms"] * 1000.0)
        f[f"psi_full_{ws}"] = _div(S["psi_full"], S["dt_ms"] * 1000.0)
        f[f"iowait_{ws}"] = _div(S["iowait"], cnt["iowait"])
        f[f"steal_{ws}"] = _div(S["steal"], cnt["steal"])
        f[f"coverage_{ws}"] = cnt["dt_ms"] * med_dt / w   # share of window with valid data
        f[f"await_volatility_ms_{ws}"] = _rolling_std(W, tick_wait_ms, w)
        f[f"throughput_volatility_mbps_{ws}"] = _rolling_std(W, tick_throughput_mbps, w)

        proc_io = W.mean(raw["proc_io_bps"], w)
        proc_write = W.mean(raw["proc_write_bps"], w)
        proc_max = W.mean(raw["proc_max_io_bps"], w)
        proc_submit = W.mean(raw["proc_submit_bps"], w)
        proc_write_submit = W.mean(raw["proc_write_submit_bps"], w)
        f[f"proc_io_bps_{ws}"] = proc_io
        f[f"proc_write_bps_{ws}"] = proc_write
        f[f"proc_max_io_bps_{ws}"] = proc_max
        f[f"proc_submit_bps_{ws}"] = proc_submit
        f[f"proc_write_submit_bps_{ws}"] = proc_write_submit
        f[f"proc_io_device_ratio_{ws}"] = _div(proc_io, f[f"bps_{ws}"])
        f[f"proc_write_device_ratio_{ws}"] = _div(proc_write, f[f"bps_{ws}"])
        f[f"proc_submit_device_ratio_{ws}"] = _div(proc_submit, f[f"bps_{ws}"])
        f[f"proc_top_share_{ws}"] = _div(proc_max, proc_io)

    w1, wl, tw = min(cfg.windows_s), max(cfg.windows_s), cfg.trend_window_s
    s1, sl = _wstr(w1), _wstr(wl)
    tsec = ts - ts[0] if n_rows else ts

    # Latency acceleration: short-window latency relative to its own recent past.
    f["lat_accel"] = _div(f[f"await_{s1}"], f[f"await_{sl}"])
    # Trends (per second) of the 1 s aggregates over the trend window.
    # Page-cache writeback pressure (gauges): the leading indicator for
    # buffered-write bursts that have not reached the device yet.
    dirty = col("mem_dirty_kb") / 1024.0
    f["dirty_mb"] = dirty
    f["writeback_mb"] = col("mem_writeback_kb") / 1024.0
    f["lat_trend"], f["tput_trend_mbps"], f["aqu_trend"], f["dirty_trend_mbps"] = W.slopes(
        [(f[f"await_{s1}"], tsec), (f[f"bps_{s1}"] / 1e6, tsec), (f[f"aqu_{s1}"], tsec),
         (dirty, tsec)], tw)
    # Throughput-scaling elasticity: d log(IOPS) / d log(in-flight). ~1 while the
    # device still scales; -> 0 (or negative) when added concurrency only buys
    # latency. This replaces "queue overflow" as the saturation signature.
    with np.errstate(invalid="ignore", divide="ignore"):
        f["scaling_elasticity"] = np.clip(
            W.slope(np.log1p(f[f"iops_{s1}"]), np.log(f[f"aqu_{s1}"] + 1e-3), tw, min_n=6), -3, 3)
    f["dirty_trend_1s_mbps"] = W.slope(dirty, tsec, w1, min_n=3)
    f["in_flight"] = col("in_flight")
    # Single-tick values (window = one sample). Too noisy to be the label, but
    # they are the finest-grained evidence of "already degraded" and are used to
    # define onset rows and the raw onset time for lead-time measurement.
    f["await_tick"] = np.where(tick_ios >= 2, tick_wait_ms, np.nan)
    f["iops_tick"] = _div(tick_ios, raw["dt_ms"] / 1000.0)
    f["latency_accel_ms_s2"] = W.slope(f["lat_trend"], tsec, tw, min_n=4)
    f["queue_accel_s2"] = W.slope(f["aqu_trend"], tsec, tw, min_n=4)
    f["throughput_accel_mbps_s2"] = W.slope(f["tput_trend_mbps"], tsec, tw, min_n=4)
    for w in cfg.windows_s:
        ws = _wstr(w)
        f[f"proc_pressure_corr_{ws}"] = _rolling_corr(
            W, f["await_1s"], raw["proc_io_bps"] / 1e6, w)

    for k, v in f.items():
        v[~np.isfinite(v)] = np.nan
    return f


def build_device_features(dev: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()
                          ) -> pd.DataFrame:
    """DataFrame wrapper: features for ONE device of ONE run, aligned with `dev`."""
    c = {k: dev[k].to_numpy() for k in RAW_INPUT_COLS if k in dev}
    return pd.DataFrame(device_features_arrays(c, cfg), index=dev.index)


def build_features_for_runs(dev_all: pd.DataFrame, device: str,
                            cfg: FeatureConfig = FeatureConfig(),
                            proc_all: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Apply build_device_features per run (windows never cross run boundaries)."""
    parts = []
    d = dev_all[dev_all["device"] == device]
    for run_id, g in d.groupby("run_id", sort=False):
        g = g.sort_values("ts")
        if proc_all is not None and not proc_all.empty:
            proc = proc_all[proc_all["run_id"] == run_id]
            rates = process_rates_at_ticks(g, proc, cfg.process_max_age_s)
            g = g.join(rates)
        F = build_device_features(g, cfg)
        F.insert(0, "run_id", run_id)
        F.insert(1, "ts", g["ts"].to_numpy())
        F.insert(2, "valid", g["valid"].to_numpy())
        if "wall" in g:
            F.insert(3, "wall", g["wall"].to_numpy())
        parts.append(F)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def feature_columns(F: pd.DataFrame) -> List[str]:
    return [c for c in F.columns if c not in META_COLS and not c.startswith("coverage_")]


@dataclass
class BaselineNormalizer:
    """Per-profile baseline, FIT ON TRAINING ROWS ONLY.

    Adds scale-free ratios (latency / baseline latency, throughput / baseline)
    so a model can transfer across storage profiles of different speeds. The
    tree model itself is scale-invariant; leakage risk is only in *where* the
    baseline is estimated, hence the explicit fit/transform split.
    """
    base_await_ms: Optional[float] = None
    base_bps: Optional[float] = None
    stats: Dict[str, float] = field(default_factory=dict)

    def fit(self, F_train: pd.DataFrame, min_iops: float = 20.0) -> "BaselineNormalizer":
        busy = F_train[F_train["iops_1s"] >= min_iops]
        src = busy if len(busy) >= 50 else F_train
        # 1 s latency median: robust to episodes (see BottleneckCriteria.fit).
        self.base_await_ms = float(np.nanmedian(src["await_1s"])) if src["await_1s"].notna().any() else 1.0
        self.base_await_ms = max(self.base_await_ms, 0.05)
        self.base_bps = float(np.nanmedian(src["bps_30s"])) if src["bps_30s"].notna().any() else 1.0
        self.base_bps = max(self.base_bps, 1.0)
        self.stats = {"base_await_ms": self.base_await_ms, "base_bps": self.base_bps,
                      "n_rows": int(len(src))}
        return self

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        F = F.copy()
        for c in [c for c in F.columns if c.startswith(("await_", "r_await_", "w_await_"))]:
            F[c + "_rel"] = F[c] / self.base_await_ms
        for c in [c for c in F.columns if c.startswith("bps_")]:
            F[c + "_rel"] = F[c] / self.base_bps
        return F
