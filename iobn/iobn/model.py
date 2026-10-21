"""Phase 3 (cont.) — Training, honest evaluation, and live inference.

Evaluation protocol (what the paper should report)
--------------------------------------------------
For the held-out split, three predictors are scored on ALL rows and on
ONSET rows (no pressure in the last 3 s):
    model          gradient-boosted trees on causal features
    persistence    "pressure now"  -> "pressure in (t+1.5, t+3.0]"
    static_util    util_1s >= 90%  (what a reactive %util alarm does)
plus episode-level metrics: share of pressure episodes warned ahead of their
start, the lead-time distribution, and false alarms per hour. A model that
merely detects existing bottlenecks matches persistence on ALL rows and gets
~0 recall on ONSET rows — the comparison makes that failure visible.
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .features import (BaselineNormalizer, FeatureConfig, build_device_features,
                       build_features_for_runs, feature_columns, aggregate_process_rates)
from .labeling import (BottleneckCriteria, LabelConfig, episodes, make_labels,
                       purged_split)


LABEL_COLS = ["state_now", "raw_now", "y", "onset", "label_ok"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_runs(run_dirs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    devs, procs, gts = [], [], []
    for d in run_dirs:
        rid = os.path.basename(os.path.normpath(d))
        dv = pd.read_csv(os.path.join(d, "device.csv"))
        dv["run_id"] = rid
        devs.append(dv)
        p = os.path.join(d, "process.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            pr = pd.read_csv(p)
            pr["run_id"] = rid
            procs.append(pr)
        g = os.path.join(d, "ground_truth.jsonl")
        if os.path.exists(g):
            gt = pd.read_json(g, lines=True)
            gt["run_id"] = rid
            gts.append(gt)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return cat(devs), cat(procs), cat(gts)


# ---------------------------------------------------------------------------
# Dataset assembly (order matters for leakage)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    device: str
    features: FeatureConfig = field(default_factory=FeatureConfig)
    criteria: BottleneckCriteria = field(default_factory=BottleneckCriteria)
    labels: LabelConfig = field(default_factory=LabelConfig)
    test_frac: float = 0.3
    val_frac: float = 0.25
    model: str = "hgb"          # "hgb" | "rf"
    seed: int = 7
    min_precision: Optional[float] = 0.80
    alarm_consecutive: int = 2
    threshold_objective: str = "recall"

    def __post_init__(self):
        if self.min_precision is not None and not 0.0 <= self.min_precision <= 1.0:
            raise ValueError("min_precision must be between 0 and 1, or None")
        if self.alarm_consecutive < 1:
            raise ValueError("alarm_consecutive must be at least 1")
        if self.threshold_objective not in {"f1", "recall"}:
            raise ValueError("threshold_objective must be 'f1' or 'recall'")


def assemble(dev_all: pd.DataFrame, cfg: TrainConfig,
             proc_all: Optional[pd.DataFrame] = None):
    """features (causal, no fitting) -> split -> fit baselines on TRAIN -> labels."""
    F = build_features_for_runs(dev_all, cfg.device, cfg.features, proc_all)
    if F.empty:
        raise ValueError(f"no rows for device {cfg.device}")
    split = purged_split(F, cfg.test_frac, cfg.labels.h_end_s, cfg.features.max_lookback_s)

    F_train = F.iloc[split.train_idx]
    crit = BottleneckCriteria(**{k: v for k, v in asdict(cfg.criteria).items()})
    if crit.baseline_await_ms is None:
        crit.fit(F_train)                       # TRAIN rows only
    norm = BaselineNormalizer().fit(F_train)    # TRAIN rows only
    F = norm.transform(F)

    labels = []
    for rid, g in F.groupby("run_id", sort=False):
        labels.append(make_labels(g, g["ts"].to_numpy(), crit, cfg.labels))
    L = pd.concat(labels).loc[F.index]
    D = pd.concat([F, L], axis=1)
    return D, split, crit, norm


def _fit_model(X: pd.DataFrame, y: np.ndarray, kind: str, seed: int):
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                   class_weight="balanced_subsample", random_state=seed)
        m.fit(X.fillna(-1.0), y)   # RF has no native NaN handling in all versions
        m._iobn_fill = -1.0
        return m
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, class_weight="balanced", random_state=seed)
    m.fit(X, y)                    # NaN handled natively (e.g. latency at idle)
    return m


def predict_proba(m, X: pd.DataFrame) -> np.ndarray:
    if getattr(m, "_iobn_fill", None) is not None:
        X = X.fillna(m._iobn_fill)
    return m.predict_proba(X)[:, 1]


def _threshold_search(y: np.ndarray, p: np.ndarray,
                      min_precision: Optional[float] = None, k: int = 2,
                      groups: Optional[np.ndarray] = None,
                      objective: str = "f1") -> Dict:
    """Choose a threshold by F1, or maximize recall subject to precision floor."""
    if objective not in {"f1", "recall"}:
        raise ValueError("objective must be 'f1' or 'recall'")
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    groups = np.zeros(len(y), dtype=int) if groups is None else np.asarray(groups)
    valid = np.isfinite(p)
    y, p, groups = y[valid], p[valid], groups[valid]
    if not len(y):
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "precision_floor_met": min_precision is None}
    candidates = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 256)))
    records = []
    for threshold in candidates:
        pred = alarm_policy_by_group(p, groups, float(threshold), k)
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        f1 = float(f1_score(y, pred, zero_division=0))
        records.append((float(threshold), precision, recall, f1))

    feasible = [r for r in records if min_precision is None or r[1] >= min_precision]
    # If the floor cannot be met on out-of-fold data, retain the best-F1 option
    # and make the unmet constraint explicit in the report.
    pool = feasible or records
    key = (lambda r: (r[3], r[1], r[2], r[0])) if objective == "f1" \
        else (lambda r: (r[2], r[1], r[3], r[0]))
    threshold, precision, recall, f1 = max(pool, key=key)
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "precision_floor_met": min_precision is None or precision >= min_precision}


def _pick_threshold(y: np.ndarray, p: np.ndarray, min_precision: Optional[float] = None,
                    k: int = 2, groups: Optional[np.ndarray] = None,
                    objective: str = "f1") -> float:
    """Compatibility wrapper returning the calibrated probability threshold."""
    return _threshold_search(y, p, min_precision, k, groups, objective)["threshold"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _row_metrics(y: np.ndarray, yhat: np.ndarray, score: Optional[np.ndarray] = None) -> Dict:
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan"),
           "precision": float(precision_score(y, yhat, zero_division=0)),
           "recall": float(recall_score(y, yhat, zero_division=0)),
           "f1": float(f1_score(y, yhat, zero_division=0))}
    if score is not None and 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    return out


def alarm_policy(p: np.ndarray, thr: float, k: int = 2) -> np.ndarray:
    """Alarm when the last k sequential scores are all >= the threshold."""
    if k < 1:
        raise ValueError("alarm debounce length must be at least 1")
    hit = np.nan_to_num(p, nan=0.0) >= thr
    if k <= 1:
        return hit
    run = np.zeros(len(hit), int)
    for i, h in enumerate(hit):
        run[i] = run[i - 1] + 1 if (h and i) else int(h)
    return run >= k


def alarm_policy_by_group(p: np.ndarray, groups: np.ndarray, thr: float,
                          k: int = 2) -> np.ndarray:
    """Apply debounce independently to each chronologically ordered run."""
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    if len(p) != len(groups):
        raise ValueError("probabilities and groups must have equal length")
    out = np.zeros(len(p), dtype=bool)
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        out[idx] = alarm_policy(p[idx], thr, k)
    return out


def _raw_onset(ts: np.ndarray, raw: np.ndarray, s: float, max_back_s: float = 3.0) -> float:
    """Walk back from the debounced episode start to the first degraded tick.

    The debounced 1 s state lags the first degraded tick; lead time must be
    measured from the earliest evidence, or the state's own lag is counted as
    "forecast". Single-tick dropouts are bridged.
    """
    i = int(np.searchsorted(ts, s))
    r = np.nan_to_num(raw, nan=0.0) > 0
    j = i
    while j - 1 >= 0 and ts[j - 1] >= s - max_back_s and (r[j - 1] or (j - 2 >= 0 and r[j - 2])):
        j -= 1
    while j < i and not r[j]:
        j += 1
    return float(min(ts[j], s)) if j < len(ts) else s


def episode_metrics(D: pd.DataFrame, alarm: np.ndarray, lcfg: LabelConfig,
                    min_quiet_s: float = 3.0, gt: Optional[pd.DataFrame] = None) -> Dict:
    """Episode-level forecasting quality.

    onset_raw = first degraded tick (see _raw_onset). An episode counts only if
    it is preceded by >= min_quiet_s without pressure. "Warned ahead" means an
    alarm in [onset_raw - h_end, onset_raw) — strictly before the device showed
    any degradation. Alarm rising edges outside episodes are split into
    near-misses (some raw degradation within h_end + 0.5 s, below the episode
    criteria) and false alarms (nothing at all).
    If ground truth is given, alarm and onset times are also reported relative
    to the start of the injected workload process, which separates "forecast
    from precursors" from "noticed the workload starting".
    """
    leads, leads_label, n_eps, n_ahead, n_false, n_near, hours = [], [], 0, 0, 0, 0, 0.0
    wl_alarm, wl_onset = [], []
    for rid, g in D.groupby("run_id", sort=False):
        ts = g["ts"].to_numpy()
        a = alarm[D.index.get_indexer(g.index)]
        raw = g["raw_now"].to_numpy() if "raw_now" in g else g["state_now"].to_numpy()
        eps = episodes(ts, g["state_now"].to_numpy())
        hours += (ts[-1] - ts[0]) / 3600.0
        G = None
        if gt is not None and not gt.empty:
            G = gt[(gt["run_id"] == rid) & gt["expected_contributor"].astype(bool)]
        prev_end = -np.inf
        for s, e in eps:
            if s - prev_end >= min_quiet_s and s - ts[0] >= lcfg.h_end_s:
                n_eps += 1
                o = _raw_onset(ts, raw, s)
                w = (ts >= o - lcfg.h_end_s) & (ts < o) & a
                wl = (ts >= s - lcfg.h_end_s) & (ts < s) & a
                if wl.any():
                    leads_label.append(float(s - ts[wl].min()))
                if w.any():
                    n_ahead += 1
                    leads.append(float(o - ts[w].min()))
                if G is not None and len(G):
                    st = G["t_start"][(G["t_start"] <= o + 0.5) & (G["t_start"] >= o - 30)]
                    if len(st):
                        ws = float(st.max())
                        wl_onset.append(o - ws)
                        aw = (ts >= ws - 5) & (ts <= o) & a
                        if aw.any():
                            wl_alarm.append(float(ts[aw].min() - ws))
            prev_end = e
        in_ep = np.zeros(len(ts), bool)
        for s, e in eps:
            in_ep |= (ts >= s) & (ts <= e)
        rising = np.flatnonzero(a & ~np.r_[False, a[:-1]])
        rr = np.nan_to_num(raw, nan=0.0) > 0
        all_starts = np.array([s for s, _ in eps])
        for i in rising:
            t = ts[i]
            if in_ep[i] or ((all_starts > t) & (all_starts <= t + lcfg.h_end_s + 0.5)).any():
                continue
            if rr[(ts > t) & (ts <= t + lcfg.h_end_s + 0.5)].any():
                n_near += 1
            else:
                n_false += 1

    q = lambda xs, p: float(np.percentile(xs, p)) if xs else float("nan")
    out = {
        "episodes_eligible": n_eps,
        "warned_ahead_of_raw_onset": n_ahead,
        "warned_ahead_frac": n_ahead / n_eps if n_eps else float("nan"),
        "lead_time_vs_raw_onset_s": {"p25": q(leads, 25), "median": q(leads, 50), "p75": q(leads, 75)},
        "lead_time_vs_debounced_label_s": {"median": q(leads_label, 50),
                                           "note": "inflated by the state's own lag; do not report"},
        "near_miss_alarms": n_near,
        "false_alarms": n_false,
        "false_alarms_per_hour": n_false / hours if hours else float("nan"),
    }
    if wl_onset:
        out["relative_to_workload_start_s"] = {
            "raw_onset_median": q(wl_onset, 50),
            "first_alarm_median": q(wl_alarm, 50),
            "note": "alarm before workload start = genuine precursor; between start and raw "
                    "onset = early recognition of a load the device has not yet felt",
        }
    return out


def evaluate(D: pd.DataFrame, p: np.ndarray, thr: float, lcfg: LabelConfig,
             gt: Optional[pd.DataFrame] = None, k: int = 2) -> Dict:
    ok = D["y"].notna().to_numpy()
    y = D["y"].to_numpy()[ok].astype(int)
    onset = D["onset"].to_numpy()[ok].astype(bool)
    pm = p[ok]
    # Strongest persistence baseline: any current evidence, debounced OR single-tick.
    now = (np.nan_to_num(D["state_now"].to_numpy()) > 0) | (np.nan_to_num(D["raw_now"].to_numpy()) > 0)
    persist = now[ok]
    static = np.nan_to_num(D["util_1s"].to_numpy()[ok]) >= 0.90

    p_full = np.full(len(D), np.nan)
    p_full[ok] = pm
    model_alarm = alarm_policy_by_group(p_full, D["run_id"].to_numpy(), thr, k)
    res = {"threshold": thr, "alarm_policy": f"{k} consecutive evaluation scores >= threshold"}
    for name, mask in (("all", np.ones_like(onset)), ("onset", onset)):
        res[name] = {
            "model": _row_metrics(y[mask], model_alarm[ok][mask], pm[mask]),
            "persistence": _row_metrics(y[mask], persist[mask]),
            "static_util90": _row_metrics(y[mask], static[mask]),
        }
    res["episodes_model"] = episode_metrics(D, model_alarm, lcfg, gt=gt)
    res["episodes_persistence"] = episode_metrics(D, now, lcfg, gt=gt)
    return res


# ---------------------------------------------------------------------------
# Train end-to-end
# ---------------------------------------------------------------------------
def train(run_dirs: List[str], cfg: TrainConfig, out_path: Optional[str] = None,
          importance: bool = True) -> Dict:
    dev_all, proc_all, gt = load_runs(run_dirs)
    D, split, crit, norm = assemble(dev_all, cfg, proc_all)
    feats = feature_columns(D.drop(columns=LABEL_COLS))

    tr = D.iloc[split.train_idx]
    te = D.iloc[split.test_idx]
    tr = tr[tr["y"].notna()]
    if tr["y"].nunique() < 2:
        raise ValueError("training labels have one class — record runs with pressure episodes "
                         "(see workloads.py) or relax BottleneckCriteria")

    # Threshold from OUT-OF-FOLD predictions inside TRAIN, never from test.
    # >= 2 training runs: leave-one-run-out (every training run is scored by a
    # model that never saw it). 1 run: one purged chronological validation cut.
    y_tr = tr["y"].astype(int).to_numpy()
    oof = np.full(len(tr), np.nan)
    train_runs = list(dict.fromkeys(tr["run_id"]))
    if len(train_runs) >= 2:
        for r in train_runs:
            hold = (tr["run_id"] == r).to_numpy()
            if len(np.unique(y_tr[~hold])) == 2:
                m_cv = _fit_model(tr.loc[~hold, feats], y_tr[~hold], cfg.model, cfg.seed)
                oof[hold] = predict_proba(m_cv, tr.loc[hold, feats])
        val_desc = f"leave-one-run-out over {len(train_runs)} training runs"
    else:
        vs = purged_split(tr.reset_index(drop=True), cfg.val_frac, cfg.labels.h_end_s,
                          cfg.features.max_lookback_s)
        if len(np.unique(y_tr[vs.train_idx])) == 2:
            m_cv = _fit_model(tr.iloc[vs.train_idx][feats], y_tr[vs.train_idx], cfg.model, cfg.seed)
            oof[vs.test_idx] = predict_proba(m_cv, tr.iloc[vs.test_idx][feats])
        val_desc = vs.description
    have = ~np.isnan(oof)
    if have.any() and len(np.unique(y_tr[have])) == 2:
        calibration_rows = tr.iloc[np.flatnonzero(have)]
        positions = calibration_rows.index.to_numpy()
        run_ids = calibration_rows["run_id"].to_numpy()
        groups = np.zeros(len(calibration_rows), dtype=int)
        segment = 0
        for i in range(1, len(groups)):
            if run_ids[i] != run_ids[i - 1] or positions[i] != positions[i - 1] + 1:
                segment += 1
            groups[i] = segment
        calibration = _threshold_search(
            y_tr[have], oof[have], cfg.min_precision, cfg.alarm_consecutive,
            groups, cfg.threshold_objective)
    else:
        calibration = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                       "precision_floor_met": cfg.min_precision is None}
    thr = calibration["threshold"]

    model = _fit_model(tr[feats], tr["y"].astype(int).to_numpy(), cfg.model, cfg.seed)
    p_te = predict_proba(model, te[feats])
    metrics = evaluate(te, p_te, thr, cfg.labels, gt=gt, k=cfg.alarm_consecutive)

    report = {
        "split": split.description,
        "threshold_selection": val_desc,
        "threshold_calibration": {
            "objective": (f"maximize debounced {cfg.threshold_objective} subject to minimum precision"
                          if cfg.min_precision is not None else
                          f"maximize debounced {cfg.threshold_objective}"),
            "minimum_precision": cfg.min_precision,
            "precision_floor_met": calibration["precision_floor_met"],
            "out_of_fold_precision": calibration["precision"],
            "out_of_fold_recall": calibration["recall"],
            "out_of_fold_f1": calibration["f1"],
            "consecutive_scores": cfg.alarm_consecutive,
        },
        "n_train": int(len(tr)), "n_test_labeled": int(te["y"].notna().sum()),
        "train_pos_rate": float(tr["y"].mean()),
        "criteria": asdict(crit), "latency_threshold_ms": crit.latency_threshold_ms,
        "normalizer": norm.stats, "label_window_s": [cfg.labels.h_start_s, cfg.labels.h_end_s],
        "n_features": len(feats), "metrics": metrics,
    }

    if importance and te["y"].notna().sum() > 200 and te["y"].nunique(dropna=True) == 2:
        from sklearn.inspection import permutation_importance
        tl = te[te["y"].notna()]
        tl = tl.sample(min(len(tl), 3000), random_state=cfg.seed)
        Xt = tl[feats] if getattr(model, "_iobn_fill", None) is None else tl[feats].fillna(-1.0)
        pi = permutation_importance(model, Xt, tl["y"].astype(int), scoring="average_precision",
                                    n_repeats=3, random_state=cfg.seed)
        order = np.argsort(-pi.importances_mean)[:12]
        report["top_features_pr_auc_drop"] = {feats[i]: round(float(pi.importances_mean[i]), 4)
                                              for i in order}

    if out_path:
        joblib.dump({"model": model, "features": feats, "threshold": thr,
                     "criteria": crit, "normalizer": norm, "feature_cfg": cfg.features,
                     "label_cfg": cfg.labels, "device": cfg.device, "report": report,
                     "alarm_consecutive": cfg.alarm_consecutive}, out_path)
    return report


# ---------------------------------------------------------------------------
# Live inference
# ---------------------------------------------------------------------------
class FastTreeScorer:
    """Single-row scoring of a fitted HistGradientBoostingClassifier in ~0.2 ms.

    sklearn calls one Cython predictor per tree (~13 us each), so a 300-tree
    model costs ~4 ms per row — fine offline, wasteful at 2-4 Hz online. Here
    all trees are packed into flat arrays and traversed together, one depth
    level per numpy step. Correctness is not assumed: the first `verify_n`
    calls are cross-checked against predict_proba, and any mismatch or
    unsupported model (categorical splits, multiclass, RF) permanently falls
    back to sklearn.
    """

    def __init__(self, model, verify_n: int = 20):
        self.model = model
        self.ok = False
        self.verify_left = verify_n
        try:
            preds = [p[0] for p in model._predictors]
            if any(len(p) != 1 for p in model._predictors) or any(
                    p.nodes["is_categorical"].any() for p in preds):
                return
            offs = np.cumsum([0] + [len(p.nodes) for p in preds[:-1]])
            nodes = np.concatenate([p.nodes for p in preds])
            self.feat = nodes["feature_idx"].astype(np.int64)
            self.thr = nodes["num_threshold"].astype(float)
            self.mleft = nodes["missing_go_to_left"].astype(bool)
            self.left = nodes["left"].astype(np.int64) + np.repeat(offs, [len(p.nodes) for p in preds])
            self.right = nodes["right"].astype(np.int64) + np.repeat(offs, [len(p.nodes) for p in preds])
            self.leaf = nodes["is_leaf"].astype(bool)
            self.value = nodes["value"].astype(float)
            self.roots = offs.astype(np.int64)
            self.base = float(np.ravel(model._baseline_prediction)[0])
            self.max_depth = int(nodes["depth"].max()) + 1
            self.ok = True
        except Exception:
            self.ok = False

    def proba(self, x: np.ndarray, X_df: pd.DataFrame) -> float:
        if not self.ok:
            return float(predict_proba(self.model, X_df)[0])
        pos = self.roots.copy()
        for _ in range(self.max_depth):
            act = ~self.leaf[pos]
            if not act.any():
                break
            p = pos[act]
            v = x[self.feat[p]]
            go_left = np.where(np.isnan(v), self.mleft[p], v <= self.thr[p])
            pos[act] = np.where(go_left, self.left[p], self.right[p])
        raw = self.base + self.value[pos].sum()
        out = 1.0 / (1.0 + np.exp(-raw))
        if self.verify_left > 0:
            self.verify_left -= 1
            ref = float(predict_proba(self.model, X_df)[0])
            if abs(ref - out) > 1e-6:
                self.ok = False
                return ref
        return float(out)


class LivePredictor:
    """Scores the newest tick using exactly the training feature code
    (device_features_arrays), on column ring buffers holding the last
    max_lookback seconds. No DataFrame churn per tick; scoring every
    `eval_every` ticks keeps inference inside the overhead budget."""

    def __init__(self, bundle_path: str, interval_s: float, eval_every: int = 2):
        from .features import RAW_INPUT_COLS
        b = joblib.load(bundle_path)
        self.b = b
        self.device = b["device"]
        self.eval_every = max(1, eval_every)
        keep = int((b["feature_cfg"].max_lookback_s + 2.0) / interval_s) + 4
        self.cols = {k: deque(maxlen=keep) for k in RAW_INPUT_COLS}
        self.n = 0
        self.feats = b["features"]
        self.scorer = FastTreeScorer(b["model"]) if getattr(b["model"], "_iobn_fill", None) is None \
            else None
        norm = b["normalizer"]
        self._rel = {f: (f[:-4], norm.base_await_ms if f.startswith(("await_", "r_await_", "w_await_"))
                         else norm.base_bps) for f in self.feats if f.endswith("_rel")}

    def push(self, dev_rows: List[Dict], proc_rows: Optional[List[Dict]] = None
             ) -> Optional[Tuple[float, float]]:
        from .features import device_features_arrays
        row = next((r for r in dev_rows if r["device"] == self.device), None)
        if row is None:
            return None
        ts = float(row["ts"])
        if proc_rows:
            rates = aggregate_process_rates(pd.DataFrame(proc_rows))
            if not rates.empty:
                latest = rates.iloc[-1]
                self._proc_rate_ts = float(latest["ts"])
                self._proc_rates = {name: float(latest[name]) for name in rates.columns
                                    if name != "ts"}
        for k, dq in self.cols.items():
            if k in ("proc_io_bps", "proc_write_bps", "proc_max_io_bps"):
                age = ts - getattr(self, "_proc_rate_ts", -np.inf)
                value = getattr(self, "_proc_rates", {}).get(k, np.nan)
                dq.append(value if age <= self.b["feature_cfg"].process_max_age_s else np.nan)
                continue
            v = row.get(k, np.nan)
            dq.append(np.nan if v in ("", None) else v)
        self.n += 1
        if self.n < 8 or self.n % self.eval_every:
            return None
        arrs = {k: np.asarray(dq, dtype=float) for k, dq in self.cols.items()}
        f = device_features_arrays(arrs, self.b["feature_cfg"])
        x = {}
        for name in self.feats:
            if name in self._rel:
                src, base = self._rel[name]
                x[name] = f[src][-1] / base
            else:
                x[name] = f[name][-1]
        if self.scorer is not None and self.scorer.ok and self.scorer.verify_left == 0:
            return float(row["ts"]), self.scorer.proba(np.array([x[k] for k in self.feats], float), None)
        X = pd.DataFrame([x], columns=self.feats)
        if self.scorer is not None:
            return float(row["ts"]), self.scorer.proba(np.array([x[k] for k in self.feats], float), X)
        return float(row["ts"]), float(predict_proba(self.b["model"], X)[0])
