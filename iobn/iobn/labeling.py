"""Phase 3 — True predictive labels and leakage-proof splits.

The leakage the review found
----------------------------
If the label at t is "is the device bottlenecked at t" (or in a window that
overlaps the feature window), the model learns to *detect* an existing
bottleneck from await_1s and scores beautifully while forecasting nothing.

What this module enforces
-------------------------
1. State definition. pressure_state(u) is computed from data in (u - W_s, u]
   only (W_s = state window, default 1 s), plus a debounce (state must hold
   for D seconds, again backward-looking). It uses *measurable* criteria:
   I/O-weighted latency vs. a training-only baseline, with an absolute floor
   and a minimum IOPS (so a single slow I/O on an idle disk is not a
   "bottleneck").

   PSI is deliberately NOT a standalone criterion (psi_mode="ignore" by
   default). io.some is "at least one task waited on I/O", which is true
   whenever ANY process runs a back-to-back synchronous I/O loop — even on a
   perfectly healthy device. In our own recordings, a PSI-OR criterion turned
   "a workload started" into a "bottleneck", and the model then "predicted"
   it by spotting rising IOPS before the 1 s / 0.5 s debounced state caught
   up. That is detection dressed up as forecasting. PSI stays available as a
   feature and as an optional AND-corroboration ("latency is up AND tasks
   are stalling").

2. Future label. y(t) = 1  iff  pressure_state(u) = 1 for some
   u in (t + h_start, t + h_end]   (default (t+1.5 s, t+3.0 s]).
   The raw data behind that label lies in (t + h_start - W_s - D, t + h_end].
   We REQUIRE h_start >= W_s + D so that interval starts strictly after t —
   the label then shares no raw samples with the features (which end at t).

3. Incomplete futures are dropped, not guessed: rows whose label window runs
   past the end of the run, or crosses a collector gap/suspend, get y = NaN.

4. Onset subset. A row is an "onset" row if there was no pressure anywhere in
   [t - L, t] — judged by BOTH the debounced state AND the single-tick raw
   state (await_tick over threshold). The raw tick state matters: the
   debounced 1 s state lags the first degraded tick by up to ~1 s, and
   without it some "onset" rows would already sit on a degraded device. On
   onset rows persistence cannot work; this is where a model must prove it
   forecasts. Metrics are always reported on ALL rows and on ONSET rows, next
   to a persistence baseline that uses the fastest evidence (raw OR state).

5. Purged splits. Hold out whole runs when there are several (preferred).
   Within one run, cut chronologically and purge both sides of the cut:
   train rows need t + h_end < cut; test rows need t - max_lookback >= cut.
   Baselines (normaliser, latency threshold) are fit on train rows only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BottleneckCriteria:
    state_window_s: float = 1.0      # must match the feature window used (await_1s)
    debounce_s: float = 0.5
    lat_multiplier: float = 3.0      # latency >= k x baseline ...
    lat_floor_ms: float = 2.0        # ... and >= absolute floor. Tune per profile:
                                     # NVMe ~0.5-1, SATA SSD ~2-5, HDD ~20-40, WSL2 vhdx: measure
    min_iops: float = 50.0
    psi_mode: str = "ignore"         # "ignore" | "and" (corroborate) | "or" (NOT recommended)
    psi_some_threshold: float = 0.10 # share of wall time some task stalled on I/O
    baseline_await_ms: Optional[float] = None   # set by fit() from TRAIN rows only

    def fit(self, F_train: pd.DataFrame) -> "BottleneckCriteria":
        """Baseline latency = median of 1 s I/O-weighted latency over busy TRAINING rows.

        Deliberately the 1 s window, not 30 s: an I/O-weighted 30 s average is
        dominated by any high-IOPS episode inside the window, so with frequent
        episodes its median drifts up to "episode latency" and the threshold
        stops detecting anything. The 1 s median stays robust while pressure
        occupies < 50% of busy time.
        """
        busy = F_train[(F_train["iops_1s"] >= self.min_iops) & F_train["await_1s"].notna()]
        if len(busy) < 20:
            busy = F_train[F_train["await_1s"].notna()]
        self.baseline_await_ms = float(busy["await_1s"].median()) if len(busy) else self.lat_floor_ms
        return self

    @property
    def latency_threshold_ms(self) -> float:
        base = self.baseline_await_ms if self.baseline_await_ms is not None else 0.0
        return max(self.lat_floor_ms, self.lat_multiplier * base)


def pressure_state(F: pd.DataFrame, crit: BottleneckCriteria, ts: np.ndarray) -> np.ndarray:
    """Backward-looking device-pressure state per row (0/1, NaN where unknown)."""
    ws = f"{int(crit.state_window_s)}s" if float(crit.state_window_s).is_integer() else f"{crit.state_window_s}s"
    lat = F[f"await_{ws}"].to_numpy()
    iops = F[f"iops_{ws}"].to_numpy()
    psi = F[f"psi_some_{ws}"].to_numpy() if f"psi_some_{ws}" in F else np.full(len(F), np.nan)
    lat_hit = (np.nan_to_num(lat, nan=0.0) >= crit.latency_threshold_ms) & (np.nan_to_num(iops) >= crit.min_iops)
    psi_hit = np.nan_to_num(psi, nan=0.0) >= crit.psi_some_threshold
    if crit.psi_mode == "and":
        hit = lat_hit & psi_hit
    elif crit.psi_mode == "or":
        hit = lat_hit | psi_hit
    else:
        hit = lat_hit
    raw = hit.astype(float)

    if crit.debounce_s > 0:
        s = pd.Series(raw, index=pd.to_timedelta(ts - ts[0], unit="s"))
        raw = np.array(s.rolling(f"{crit.debounce_s}s", min_periods=1).min().to_numpy(), dtype=float)
    valid = F["valid"].astype(bool).to_numpy() if "valid" in F else np.ones(len(F), bool)
    raw[~valid] = np.nan
    return raw


def raw_tick_state(F: pd.DataFrame, crit: BottleneckCriteria) -> np.ndarray:
    """Single-tick evidence of degradation (no window, no debounce)."""
    lat = np.nan_to_num(F["await_tick"].to_numpy(), nan=0.0) if "await_tick" in F else np.zeros(len(F))
    iops = np.nan_to_num(F["iops_tick"].to_numpy(), nan=0.0) if "iops_tick" in F else np.zeros(len(F))
    r = ((lat >= crit.latency_threshold_ms) & (iops >= crit.min_iops)).astype(float)
    valid = F["valid"].astype(bool).to_numpy() if "valid" in F else np.ones(len(F), bool)
    r[~valid] = np.nan
    return r


def _window_count(ts: np.ndarray, x: np.ndarray, lo_off: float, hi_off: float,
                  left_open: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """For each i: sum and count of x[j] with ts[j] in (ts[i]+lo_off, ts[i]+hi_off]."""
    side_lo = "right" if left_open else "left"
    lo = np.searchsorted(ts, ts + lo_off, side=side_lo)
    hi = np.searchsorted(ts, ts + hi_off, side="right")
    xv = np.nan_to_num(x, nan=0.0)
    cs = np.concatenate([[0.0], np.cumsum(xv)])
    return cs[hi] - cs[lo], (hi - lo).astype(float)


@dataclass
class LabelConfig:
    h_start_s: float = 1.5
    h_end_s: float = 3.0
    onset_lookback_s: float = 3.0
    min_future_coverage: float = 0.8   # share of expected samples present in the label window


def make_labels(F: pd.DataFrame, ts: np.ndarray, crit: BottleneckCriteria,
                lcfg: LabelConfig = LabelConfig(), interval_s: Optional[float] = None
                ) -> pd.DataFrame:
    """Labels for ONE run. Returns columns: state_now, raw_now, y, onset, label_ok."""
    if lcfg.h_start_s < crit.state_window_s + crit.debounce_s:
        raise ValueError(
            f"leakage: h_start ({lcfg.h_start_s}s) < state window + debounce "
            f"({crit.state_window_s + crit.debounce_s}s) — label would reuse raw data <= t")
    ts = np.asarray(ts, float)
    state = pressure_state(F, crit, ts)
    unknown = np.isnan(state).astype(float)

    pos, n = _window_count(ts, state, lcfg.h_start_s, lcfg.h_end_s)
    unk, _ = _window_count(ts, unknown, lcfg.h_start_s, lcfg.h_end_s)
    if interval_s is None:
        interval_s = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    expected = (lcfg.h_end_s - lcfg.h_start_s) / interval_s
    # Also need the *state* inputs for the label window to be valid, which
    # reach back W_s + D before h_start.
    reach_lo = lcfg.h_start_s - crit.state_window_s - crit.debounce_s
    unk_inputs, _ = _window_count(ts, unknown, reach_lo, lcfg.h_end_s)
    label_ok = (n >= lcfg.min_future_coverage * expected) & (unk == 0) & (unk_inputs == 0)
    label_ok &= (ts + lcfg.h_end_s) <= ts[-1]

    y = np.where(label_ok, (pos > 0).astype(float), np.nan)

    # onset: no pressure anywhere in [t - L, t], debounced OR raw tick-level
    raw = raw_tick_state(F, crit)
    past_pos, _ = _window_count(ts, state, -lcfg.onset_lookback_s, 0.0, left_open=False)
    past_raw, _ = _window_count(ts, raw, -lcfg.onset_lookback_s, 0.0, left_open=False)
    onset = (past_pos == 0) & (past_raw == 0) & ~np.isnan(state)

    return pd.DataFrame({"state_now": state, "raw_now": raw, "y": y, "onset": onset.astype(int),
                         "label_ok": label_ok.astype(int)}, index=F.index)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
@dataclass
class SplitPlan:
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str


def purged_split(D: pd.DataFrame, test_frac: float, h_end_s: float, max_lookback_s: float,
                 test_runs: Optional[List[str]] = None) -> SplitPlan:
    """Leakage-safe train/test split.

    * >= 2 runs: hold out whole runs (latest by start time unless given).
      Runs are separate recordings, so no purge is needed between them.
    * 1 run: chronological cut with a two-sided purge zone.
    """
    runs = list(dict.fromkeys(D["run_id"]))
    if len(runs) >= 2:
        if test_runs is None:
            # Latest recordings go to test (wall-clock start if available).
            key = "wall" if "wall" in D else "ts"
            order = list(D.groupby("run_id")[key].min().sort_values().index)
            k = max(1, int(round(test_frac * len(runs))))
            test_runs = order[-k:]
        mask = D["run_id"].isin(test_runs).to_numpy()
        return SplitPlan(np.flatnonzero(~mask), np.flatnonzero(mask),
                         f"run holdout: test runs = {test_runs}")

    ts = D["ts"].to_numpy()
    t0, t1 = ts.min(), ts.max()
    cut = t0 + (1 - test_frac) * (t1 - t0)
    train = ts + h_end_s < cut
    test = ts - max_lookback_s >= cut
    dropped = int((~train & ~test).sum())
    return SplitPlan(np.flatnonzero(train), np.flatnonzero(test),
                     f"chronological cut at t={cut:.1f}s, purged {dropped} rows "
                     f"(h_end={h_end_s}s before, lookback={max_lookback_s}s after)")


def purged_walk_forward(ts: np.ndarray, n_folds: int, h_end_s: float, max_lookback_s: float,
                        min_train_frac: float = 0.4) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV within one time-ordered block (for threshold/hparam tuning)."""
    t0, t1 = ts.min(), ts.max()
    span = t1 - t0
    edges = np.linspace(t0 + min_train_frac * span, t1, n_folds + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.flatnonzero(ts + h_end_s < a)
        te = np.flatnonzero((ts - max_lookback_s >= a) & (ts + h_end_s <= b))
        if len(tr) and len(te):
            yield tr, te


def episodes(ts: np.ndarray, state: np.ndarray, min_gap_s: float = 1.0) -> List[Tuple[float, float]]:
    """Contiguous pressure episodes as (start_ts, end_ts); merges gaps < min_gap_s."""
    s = np.nan_to_num(state, nan=0.0) > 0
    out: List[Tuple[float, float]] = []
    i, n = 0, len(s)
    while i < n:
        if s[i]:
            j = i
            while j + 1 < n and s[j + 1]:
                j += 1
            if out and ts[i] - out[-1][1] < min_gap_s:
                out[-1] = (out[-1][0], ts[j])
            else:
                out.append((ts[i], ts[j]))
            i = j + 1
        else:
            i += 1
    return out
