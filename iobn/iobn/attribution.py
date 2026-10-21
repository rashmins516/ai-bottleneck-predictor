"""Phase 4 — "Likely contributor" attribution, with cross-validation.

Why bytes != blame
------------------
* write_bytes in /proc/[pid]/io is charged when a page is DIRTIED, not when
  it reaches the device. Writeback happens later, in kernel flusher threads,
  possibly merged with other processes' pages. A process can show 2 GB of
  write_bytes during a period of zero device writes, and vice versa.
* cancelled_write_bytes: dirtied pages that were truncated before writeback
  (e.g. temp files deleted) — never hit the device.
* read_bytes counts reads that went to the block layer. Page-cache hits do
  not count — but rchar does (and rchar also counts pipes, sockets, ttys).
* Small synchronous I/O (fsync/O_DSYNC, 4 KiB random) hurts latency far more
  than its byte share suggests: a process doing 1% of the bytes can cause
  most of the queueing. Journal commits (jbd2) triggered by one process's
  fsyncs are executed by a kernel thread.
* Processes that block on I/O (delayacct_blkio_ticks) are usually VICTIMS.
  We report them separately and never use delay as evidence of guilt.

So the module ranks processes on several independent signals, states its
confidence, reports the unattributed residual, and phrases every output as
"likely contributors". validate_against_ground_truth() measures how often
the top-ranked process is the one a controlled workload deliberately made
misbehave — that hit rate, not the ranking itself, is what the paper can claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SECTOR = 512
# Kernel threads that execute I/O on behalf of others. Shown as context, never ranked.
KERNEL_PROXY_PREFIXES = ("kworker", "jbd2", "kswapd", "writeback", "flush-", "xfsaild",
                         "btrfs", "kcryptd", "dmcrypt", "blkcg_punt", "loop", "nvme",
                         "scsi", "md", "dm-", "kcompactd", "khugepaged")


@dataclass
class AttributionConfig:
    lookback_s: float = 10.0
    max_lag_ticks: int = 2           # process series may LEAD device latency by up to N process ticks
    w_phys: float = 0.5              # share of attributed physical bytes
    w_ops: float = 0.2               # share of estimated physical operations (small-I/O storms)
    w_coupling: float = 0.3          # rank correlation of process rate with device latency
    min_points_coupling: int = 5
    top_k: int = 5
    min_share: float = 0.02          # below this share of bytes AND ops a process is not listed
    exclude_pids: tuple = ()


@dataclass
class Contributor:
    pid: int
    start_ticks: int
    comm: str
    cmdline: str
    phys_bytes: float
    phys_share: float
    est_ops_share: float
    coupling_rho: Optional[float]
    coupling_lag_ticks: Optional[int]
    cache_ratio: Optional[float]        # physical / logical bytes; low => mostly page-cache served
    reaped_children: bool
    score: float


@dataclass
class AttributionReport:
    t_end: float
    device: str
    window_s: float
    device_bytes: float
    attributed_bytes: float
    residual_frac: float                # share of device bytes not explained by process counters
    confidence: str                     # high | medium | low
    contributors: List[Contributor]
    likely_victims: List[Tuple[int, str, float]]   # (pid, comm, blkio delay seconds)
    kernel_proxies: List[Tuple[int, str, float]]   # (pid, comm, bytes)
    notes: List[str] = field(default_factory=list)

    def to_log(self, prob: Optional[float] = None, horizon: Tuple[float, float] = (1.5, 3.0)) -> str:
        hdr = (f"[iobn] predicted device pressure on {self.device}"
               + (f" (p={prob:.2f}, horizon {horizon[0]}-{horizon[1]} s)" if prob is not None else "")
               + f" — LIKELY CONTRIBUTORS over last {self.window_s:.0f} s "
               f"[attribution confidence: {self.confidence}]")
        lines = [hdr]
        for i, c in enumerate(self.contributors, 1):
            coup = (f", latency coupling rho={c.coupling_rho:+.2f} (lead {c.coupling_lag_ticks} tick)"
                    if c.coupling_rho is not None else "")
            extra = " [includes I/O of reaped children]" if c.reaped_children else ""
            lines.append(f"  {i}. pid {c.pid} '{c.comm}' — {c.phys_share:.0%} of attributed physical "
                         f"bytes, {c.est_ops_share:.0%} of est. ops{coup}{extra}"
                         + (f"\n       cmd: {c.cmdline}" if c.cmdline else ""))
        if self.device_bytes > 0 and self.residual_frac >= 0:
            lines.append(f"  unattributed device bytes: {self.residual_frac:.0%}")
        elif self.device_bytes > 0:
            lines.append(f"  processes were charged {self.attributed_bytes / self.device_bytes:.1f}x the bytes "
                         "the device moved: data is still in the page cache, writeback pending")
        else:
            lines.append("  device transferred ~nothing in this window yet: ranking reflects page-cache "
                         "dirtying / submissions AHEAD of writeback")
        if self.likely_victims:
            v = ", ".join(f"{p} '{n}' ({s:.1f}s blocked)" for p, n, s in self.likely_victims[:3])
            lines.append(f"  most delayed (likely victims, not causes): {v}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        lines.append("  Attribution is correlational: these processes are candidates for "
                     "throttling review, not proven causes.")
        return "\n".join(lines)


def _is_proxy(comm: str, kthread: int) -> bool:
    return bool(kthread) or comm.startswith(KERNEL_PROXY_PREFIXES)


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return None
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_likely_contributors(proc: pd.DataFrame, dev: pd.DataFrame, device: str, t_end: float,
                             cfg: AttributionConfig = AttributionConfig(),
                             cmdline_lookup=None) -> AttributionReport:
    """Rank processes over (t_end - lookback, t_end]. `proc`/`dev` are one run's rows."""
    t0 = t_end - cfg.lookback_s
    P = proc[(proc["ts"] > t0) & (proc["ts"] <= t_end)].copy()
    Dv = dev[(dev["device"] == device) & (dev["ts"] > t0) & (dev["ts"] <= t_end)]
    Dv = Dv[Dv["valid"].astype(bool)]
    dev_bytes = float(((Dv["d_rd_sectors"] + Dv["d_wr_sectors"]) * SECTOR).sum())

    notes: List[str] = []
    if P.empty:
        return AttributionReport(t_end, device, cfg.lookback_s, dev_bytes, 0.0, 1.0, "low", [], [], [],
                                 ["no process I/O observed in window"])

    for c in ("d_read_bytes", "d_write_bytes", "d_cancelled_write_bytes", "d_rchar", "d_wchar",
              "d_syscr", "d_syscw"):
        P[c] = pd.to_numeric(P[c], errors="coerce").fillna(0.0)
    P["phys"] = (P["d_read_bytes"] + P["d_write_bytes"] - P["d_cancelled_write_bytes"]).clip(lower=0)
    P["logical"] = P["d_rchar"] + P["d_wchar"]
    P["ops"] = P["d_syscr"] + P["d_syscw"]
    P["blk"] = pd.to_numeric(P.get("d_blkio_ticks"), errors="coerce").fillna(0.0)
    P["proxy"] = [_is_proxy(str(c), int(k)) for c, k in zip(P["comm"], P["kthread"])]

    key = ["pid", "start_ticks"]
    agg = P.groupby(key).agg(comm=("comm", "last"), phys=("phys", "sum"), logical=("logical", "sum"),
                             ops=("ops", "sum"), blk=("blk", "sum"), proxy=("proxy", "max"),
                             reaped=("reaped_child_adjusted", "max")).reset_index()

    if cfg.exclude_pids:
        agg = agg[~agg["pid"].isin(cfg.exclude_pids)]   # e.g. the collector itself
    proxies = agg[agg["proxy"]]
    users = agg[~agg["proxy"]].copy()
    tick_s = 1.0
    try:
        from .telemetry import CLK_TCK
        tick_s = 1.0 / CLK_TCK
    except Exception:
        pass
    victims = (users[users["blk"] > 0].sort_values("blk", ascending=False)
               .head(5)[["pid", "comm", "blk"]])
    victims_l = [(int(r.pid), str(r.comm), float(r.blk) * tick_s) for r in victims.itertuples()]

    attributed = float(users["phys"].sum())
    residual = 1.0 - attributed / dev_bytes if dev_bytes > 0 else float("nan")

    # Estimated physical operations: syscalls scaled by the physical/logical
    # byte ratio, capped at phys_bytes / 4 KiB (a physical I/O moves at least
    # one page, so 8 KiB of physical bytes cannot be more than 2 device ops).
    # Crude, but it surfaces small synchronous I/O storms that bytes hide,
    # without promoting chatty loggers that barely touch the disk.
    ratio = (users["phys"] / users["logical"].where(users["logical"] > 0)).clip(upper=1.0)
    users["cache_ratio"] = ratio
    users["est_ops"] = np.minimum(users["ops"] * ratio.fillna(0.0), users["phys"] / 4096.0)
    users = users[(users["phys"] > 0) | (users["est_ops"] > 0)]
    if users.empty:
        notes.append("device I/O in window is not attributable to user processes "
                     "(kernel writeback / metadata / other VMs)")
        return AttributionReport(t_end, device, cfg.lookback_s, dev_bytes, 0.0,
                                 1.0 if dev_bytes > 0 else float("nan"), "low", [], victims_l,
                                 [(int(r.pid), str(r.comm), float(r.phys)) for r in proxies.itertuples()],
                                 notes)
    users["phys_share"] = users["phys"] / max(users["phys"].sum(), 1.0)
    users["ops_share"] = users["est_ops"] / max(users["est_ops"].sum(), 1.0)
    # Coupling alone never qualifies a process: it must own a non-trivial share.
    users = users[(users["phys_share"] >= cfg.min_share) | (users["ops_share"] >= cfg.min_share)]

    # Temporal coupling: process physical rate vs device latency on the
    # process-tick grid; process allowed to LEAD by 0..max_lag ticks.
    tticks = np.sort(P["ts"].unique())
    dev_lat = np.full(len(tticks), np.nan)
    prev = t0
    for i, tk in enumerate(tticks):
        w = Dv[(Dv["ts"] > prev) & (Dv["ts"] <= tk)]
        n = (w["d_rd_ios"] + w["d_wr_ios"]).sum()
        if n > 0:
            dev_lat[i] = (w["d_rd_ticks"] + w["d_wr_ticks"]).sum() / n
        prev = tk
    rho_l, lag_l = [], []
    for r in users.itertuples():
        s = P[(P["pid"] == r.pid) & (P["start_ticks"] == r.start_ticks)].set_index("ts")["phys"]
        series = s.reindex(tticks).fillna(0.0).to_numpy()
        best, best_lag = None, None
        if (series > 0).sum() >= 2 and len(tticks) >= cfg.min_points_coupling:
            for lag in range(cfg.max_lag_ticks + 1):
                a = series[: len(series) - lag] if lag else series
                b = dev_lat[lag:]
                rho = _spearman(a, b)
                if rho is not None and (best is None or rho > best):
                    best, best_lag = rho, lag
        rho_l.append(best)
        lag_l.append(best_lag)
    users["rho"], users["lag"] = rho_l, lag_l

    users["score"] = (cfg.w_phys * users["phys_share"] + cfg.w_ops * users["ops_share"]
                      + cfg.w_coupling * users["rho"].astype(float).fillna(0.0).clip(lower=0.0))
    users = users.sort_values("score", ascending=False)

    top = users.head(cfg.top_k)
    contributors = [Contributor(
        pid=int(r.pid), start_ticks=int(r.start_ticks), comm=str(r.comm),
        cmdline=(cmdline_lookup(int(r.pid)) if cmdline_lookup else ""),
        phys_bytes=float(r.phys), phys_share=float(r.phys_share), est_ops_share=float(r.ops_share),
        coupling_rho=None if pd.isna(r.rho) else float(r.rho),
        coupling_lag_ticks=None if pd.isna(r.lag) else int(r.lag),
        cache_ratio=None if pd.isna(r.cache_ratio) else float(r.cache_ratio),
        reaped_children=bool(r.reaped), score=float(r.score)) for r in top.itertuples()]

    # Confidence: agreement of independent signals + how much is explained.
    conf = "low"
    if contributors:
        c1 = contributors[0]
        s2 = contributors[1].score if len(contributors) > 1 else 0.0
        separated = c1.score >= 1.5 * s2 if s2 > 0 else True
        coupled = c1.coupling_rho is not None and c1.coupling_rho >= 0.4
        explained = not np.isnan(residual) and abs(residual) <= 0.5
        # A ranking that explains < 20% of what the device did is not worth more than "low".
        explains_some = np.isnan(residual) or residual <= 0.8
        if separated and coupled and explained and c1.phys_share >= 0.4:
            conf = "high"
        elif separated and (coupled or c1.phys_share >= 0.5) and explains_some:
            conf = "medium"

    dirty_writes = float(users["phys"].sum())
    if dev_bytes > 0 and dirty_writes > 1.5 * dev_bytes:
        notes.append("processes dirtied more bytes than the device transferred: buffered writes are "
                     "still in the page cache; device pressure may arrive later (writeback)")
    if not np.isnan(residual) and residual > 0.5:
        notes.append("over half of device bytes are unattributed (writeback of earlier dirtying, "
                     "journal/metadata, exited processes, or I/O from outside this kernel)")
    if any(c.reaped_children for c in contributors):
        notes.append("a parent's counters absorbed exited children; check its child workers")
    if not proxies.empty and proxies["phys"].sum() > 0:
        notes.append("kernel threads performed I/O on behalf of user processes (not ranked)")

    return AttributionReport(t_end, device, cfg.lookback_s, dev_bytes, attributed,
                             float(residual) if not np.isnan(residual) else float("nan"), conf,
                             contributors, victims_l,
                             [(int(r.pid), str(r.comm), float(r.phys)) for r in proxies.itertuples()],
                             notes)


def attribute_predicted_bottleneck_json(
        proc: pd.DataFrame, dev: pd.DataFrame, device: str, t_end: float,
        predicted_probability: float, threshold: float,
        cfg: AttributionConfig = AttributionConfig()) -> Optional[str]:
    """Return JSON attribution for an active prediction, or None below threshold.

    `proc` and `dev` must contain recent rows from the collector's process
    tracker and device sampler. A one-time read of /proc/[pid]/io contains
    lifetime counters; it cannot be compared fairly with a device interval.
    The tracked interval deltas are therefore used for contribution estimates,
    while this function refreshes the active-process inventory for command paths.

    delayacct_blkio_ticks is reported only as a blocked-process (victim) signal.
    Linux does not expose per-process block-layer wait time in /proc/[pid]/io.
    """
    if not 0.0 <= predicted_probability <= 1.0:
        raise ValueError("predicted_probability must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if predicted_probability < threshold:
        return None

    from . import telemetry as T

    active, scan_stats = T.scan_processes()
    active_keys = {(key.pid, key.start_ticks) for key in active}

    def executable_path(pid: int, start_ticks: int) -> Optional[str]:
        if (pid, start_ticks) not in active_keys:
            return None
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None

    report = rank_likely_contributors(proc, dev, device, t_end, cfg)
    t0 = t_end - cfg.lookback_s
    window = proc[(proc["ts"] > t0) & (proc["ts"] <= t_end)].copy()
    for col in ("d_read_bytes", "d_write_bytes", "d_cancelled_write_bytes", "d_blkio_ticks"):
        if col not in window:
            window[col] = 0.0
        window[col] = pd.to_numeric(window[col], errors="coerce").fillna(0.0)

    window["read_bytes"] = window["d_read_bytes"].clip(lower=0)
    window["write_bytes"] = (
        window["d_write_bytes"] - window["d_cancelled_write_bytes"]
    ).clip(lower=0)
    window["block_wait_ticks"] = window["d_blkio_ticks"].clip(lower=0)
    by_process = (window.groupby(["pid", "start_ticks"], as_index=False)
                  .agg(comm=("comm", "last"),
                       read_bytes=("read_bytes", "sum"),
                       write_bytes=("write_bytes", "sum"),
                       block_wait_ticks=("block_wait_ticks", "sum")))

    read_total = float(by_process["read_bytes"].sum())
    write_total = float(by_process["write_bytes"].sum())
    wait_total = float(by_process["block_wait_ticks"].sum())
    try:
        tick_hz = float(T.CLK_TCK)
    except (AttributeError, TypeError, ValueError):
        tick_hz = 0.0

    result = asdict(report)
    contributors = []
    for candidate in report.contributors:
        match = by_process[
            (by_process["pid"] == candidate.pid)
            & (by_process["start_ticks"] == candidate.start_ticks)
        ]
        row = match.iloc[0] if not match.empty else None
        read_bytes = float(row["read_bytes"]) if row is not None else 0.0
        write_bytes = float(row["write_bytes"]) if row is not None else 0.0
        wait_ticks = float(row["block_wait_ticks"]) if row is not None else 0.0
        contributors.append({
            "pid": candidate.pid,
            "start_ticks": candidate.start_ticks,
            "command": candidate.comm,
            "command_path": executable_path(candidate.pid, candidate.start_ticks),
            "active_at_attribution": (candidate.pid, candidate.start_ticks) in active_keys,
            "read_bytes": read_bytes,
            "read_share_pct": 100.0 * read_bytes / read_total if read_total else 0.0,
            "write_bytes": write_bytes,
            "write_share_pct": 100.0 * write_bytes / write_total if write_total else 0.0,
            "physical_bytes": candidate.phys_bytes,
            "physical_share_pct": 100.0 * candidate.phys_share,
            "block_wait_seconds": wait_ticks / tick_hz if tick_hz else None,
            "block_wait_share_pct": 100.0 * wait_ticks / wait_total if wait_total else None,
            "latency_coupling_rho": candidate.coupling_rho,
            "score": candidate.score,
            "reaped_children_included": candidate.reaped_children,
        })

    result["prediction"] = {
        "probability": float(predicted_probability),
        "threshold": float(threshold),
        "triggered": True,
    }
    result["contributors"] = contributors
    result["block_wait_semantics"] = (
        "delayacct_blkio_ticks indicates time blocked on I/O (victim evidence), "
        "not process-caused block-layer wait."
    )
    result["process_scan"] = {
        "active_processes": len(active),
        "io_denied": scan_stats.io_denied,
        "vanished": scan_stats.vanished,
        "malformed": scan_stats.malformed,
    }
    return json.dumps(_json_safe(result), indent=2, allow_nan=False, default=_json_value)


def _json_value(value):
    """Convert non-finite numpy scalars to JSON null without hiding bad types."""
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Cross-validation against known workloads
# ---------------------------------------------------------------------------
def validate_against_ground_truth(proc_all: pd.DataFrame, dev_all: pd.DataFrame, gt: pd.DataFrame,
                                  device: str, query_times: Dict[str, Sequence[float]],
                                  cfg: AttributionConfig = AttributionConfig()) -> Dict:
    """Score attribution where the answer is known.

    gt rows (from workloads.py): run_id, role, pid, start_ticks, t_start, t_end,
    expected_contributor (bool). For each query time (e.g. every model alarm,
    or every pressure-episode start) in a run:
      * culprit windows: an expected contributor was active -> hit@1, hit@3, MRR,
        and the rate of WRONG answers given with high/medium confidence
      * decoy windows: how often the page-cache decoy (huge rchar, ~0 physical
        reads) is ranked in the top 3 — should be ~0 if attribution is cache-aware
      * control windows: no culprit active -> who gets ranked (expect the
        background process; attribution is only run on alarms in production)
    hit@1 is also broken down by confidence label, which tells you whether
    "high" actually means anything.
    """
    rows = []
    for rid, times in query_times.items():
        P = proc_all[proc_all["run_id"] == rid]
        Dv = dev_all[dev_all["run_id"] == rid]
        G = gt[gt["run_id"] == rid] if not gt.empty else gt
        role_of = {(int(r.pid), int(r.start_ticks)): r.role for r in G.itertuples()
                   if r.start_ticks == r.start_ticks} if not G.empty else {}
        for t in times:
            rep = rank_likely_contributors(P, Dv, device, t, cfg)
            active = G[(G["t_start"] <= t) & (G["t_end"] >= t - cfg.lookback_s * 0.5)] if not G.empty else G
            cul = active[active["expected_contributor"].astype(bool)] if not active.empty else active
            culprits = set(zip(cul["pid"], cul["start_ticks"])) if not cul.empty else set()
            decoys = set(zip(active["pid"], active["start_ticks"])) if not active.empty else set()
            decoys = {k for k in decoys if role_of.get(k) == "decoy"}
            ranked = [(c.pid, c.start_ticks) for c in rep.contributors]
            rank = next((i + 1 for i, k in enumerate(ranked) if k in culprits), None)
            roles = sorted({role_of.get(k, "?") for k in culprits})
            rows.append({"run_id": rid, "t": t, "has_culprit": bool(culprits),
                         "culprit_roles": "+".join(roles) if roles else "",
                         "rank": rank, "confidence": rep.confidence,
                         "residual": rep.residual_frac,
                         "top1_role": role_of.get(ranked[0], "untracked") if ranked else None,
                         "decoy_active": bool(decoys),
                         "decoy_top3": any(k in decoys for k in ranked[:3])})
    R = pd.DataFrame(rows)
    if R.empty:
        return {"n": 0}
    C = R[R["has_culprit"]]
    N = R[~R["has_culprit"]]
    Dc = R[R["decoy_active"]]
    rr = C["rank"].map(lambda r: 0.0 if pd.isna(r) else 1.0 / r) if len(C) else pd.Series(dtype=float)
    out = {
        "n_queries": int(len(R)),
        "culprit_windows": int(len(C)),
        "hit_at_1": float((C["rank"] == 1).mean()) if len(C) else float("nan"),
        "hit_at_3": float((C["rank"] <= 3).mean()) if len(C) else float("nan"),
        "mrr": float(rr.mean()) if len(C) else float("nan"),
        # wrong answer stated with confidence: the failure that matters operationally
        "wrong_confident_rate": float(((C["rank"] != 1) & C["confidence"].isin(["high", "medium"])).mean())
        if len(C) else float("nan"),
        "hit_at_1_by_confidence": {k: round(float((g["rank"] == 1).mean()), 3)
                                   for k, g in C.groupby("confidence")} if len(C) else {},
        "confidence_counts": C["confidence"].value_counts().to_dict() if len(C) else {},
        "hit_at_1_by_culprit_role": {k: {"n": int(len(g)), "hit_at_1": round(float((g["rank"] == 1).mean()), 3)}
                                     for k, g in C.groupby("culprit_roles")} if len(C) else {},
        "decoy_windows": int(len(Dc)),
        "decoy_in_top3_rate": float(Dc["decoy_top3"].mean()) if len(Dc) else float("nan"),
        "control_windows": int(len(N)),
        "control_top1_roles": N["top1_role"].value_counts().to_dict() if len(N) else {},
        "median_residual": float(R["residual"].median()),
    }
    return out
