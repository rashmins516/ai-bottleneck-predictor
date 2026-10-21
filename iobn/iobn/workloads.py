"""Reproducible workload generator with ground truth (drives Phases 3–5).

Every heavy event runs in its OWN short-lived process, so the run exercises
the collector's churn handling and the reaped-child accounting, and every
event is logged to ground_truth.jsonl with (pid, start_ticks, t_start, t_end,
role, expected_contributor). Timestamps use CLOCK_MONOTONIC, the same clock
as the collector.

Roles
-----
bg       light rate-limited random I/O for the whole run   (not a contributor)
burst    buffered write burst, optionally fsync'd; Dirty pages build up before
         writeback — a genuinely predictable onset            (contributor)
db       N workers of mixed random O_DIRECT reads + writes with periodic
         fdatasync, variable block sizes                      (contributor)
rogue    one unthrottled O_DIRECT 4 KiB random read/write + fdatasync loop —
         few bytes, many synchronous ops                       (contributor)
decoy    sequential re-reads of a page-cached file: huge rchar, ~0 read_bytes
         (NOT a contributor; tests that attribution is cache-aware)

Heavy events can ramp up over `ramp_s` (rate grows from 10% to 100%). Real
bottlenecks often ramp; abrupt ones (ramp_s = 0) are inherently hard to see
1.5 s ahead, and the evaluation reports them honestly.

Pitfalls handled
----------------
* Files are pre-filled with real data, never posix_fallocate()d: O_DIRECT
  reads of unwritten extents return zeros without touching the device.
* O_DIRECT needs aligned buffers (anonymous mmap is page-aligned) and is
  unsupported on tmpfs/9p (/mnt/c on WSL): we fall back to buffered I/O +
  fsync + posix_fadvise(DONTNEED) and record that in the ground truth.
"""
from __future__ import annotations

import json
import mmap
import multiprocessing as mp
import os
import random
import signal
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .telemetry import parse_proc_stat, read_file

BS_CHOICES = (4096, 16384, 65536)
CONTRIBUTOR_ROLES = {"burst", "db", "rogue"}


def _set_comm(name: str):
    try:
        with open("/proc/self/comm", "w") as f:
            f.write(name[:15])
    except OSError:
        pass


def _aligned(size: int) -> mmap.mmap:
    m = mmap.mmap(-1, size)
    m.write(os.urandom(min(size, 65536)) * (size // min(size, 65536)))
    m.seek(0)
    return m


def _open(path: str, flags: int, direct: bool):
    if direct:
        try:
            return os.open(path, flags | os.O_DIRECT, 0o644), True
        except OSError:
            pass  # EINVAL: filesystem without O_DIRECT (tmpfs, 9p, some overlays)
    return os.open(path, flags, 0o644), False


def prefill(path: str, size_mb: int):
    """Write real data so later reads hit the device (no unwritten extents)."""
    if os.path.exists(path) and os.path.getsize(path) >= size_mb << 20:
        return
    chunk = os.urandom(1 << 20)
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    _drop_cache(path)


def _drop_cache(path: str):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # clean pages only; no root needed
        os.close(fd)
    except OSError:
        pass


def _rate(t: float, t0: float, ramp_s: float, peak: float) -> float:
    if ramp_s <= 0:
        return peak
    return peak * min(1.0, 0.1 + 0.9 * (t - t0) / ramp_s)


# --------------------------------------------------------------------------
# Worker bodies (each runs in its own process until SIGTERM or `seconds`)
# --------------------------------------------------------------------------
def _install_term():
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    return stop


def w_random_io(path: str, seconds: float, peak_ops: float, read_frac: float, ramp_s: float,
                bs_choices=BS_CHOICES, sync_every: int = 8, seed: int = 0, comm: str = "iobn-db",
                threads: int = 1):
    """Random I/O from ONE process with `threads` concurrent submitters.

    Threads share the PID, so /proc/[pid]/io aggregates them — this is how a
    single misbehaving daemon reaches a high queue depth. Python releases the
    GIL inside preadv/pwritev/fdatasync, so the concurrency is real.
    Ramp: with a rate limit, the rate ramps 10% -> 100% over ramp_s; without
    one (peak_ops = 0), thread starts are staggered over ramp_s, i.e.
    offered concurrency ramps.
    """
    import threading
    _set_comm(comm)
    stop = _install_term()
    size = os.path.getsize(path)
    fd, direct = _open(path, os.O_RDWR, True)
    t0 = time.monotonic()

    def loop(ti: int):
        rnd = random.Random(seed * 1000 + ti)
        bufs = {bs: _aligned(bs) for bs in bs_choices}
        if peak_ops <= 0 and ramp_s > 0 and threads > 1:
            time.sleep(ramp_s * ti / (threads - 1))
        per_thread = peak_ops / threads if peak_ops > 0 else 0.0
        nxt, writes = time.monotonic(), 0
        while not stop["v"] and time.monotonic() - t0 < seconds:
            now = time.monotonic()
            r = _rate(now, t0, ramp_s, per_thread) if per_thread > 0 else 0
            if r > 0:
                nxt += 1.0 / r
                if nxt > now:
                    time.sleep(nxt - now)
                elif now - nxt > 1.0:
                    nxt = now  # don't accumulate unbounded debt
            bs = rnd.choice(bs_choices)
            off = rnd.randrange(0, size - bs) // 4096 * 4096
            buf = bufs[bs]
            if rnd.random() < read_frac:
                os.preadv(fd, [buf], off)
            else:
                os.pwritev(fd, [buf], off)
                writes += 1
                if writes % sync_every == 0:
                    os.fdatasync(fd)
                    if not direct:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    ts = [threading.Thread(target=loop, args=(i,), daemon=True) for i in range(threads)]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts) and not stop["v"]:
        time.sleep(0.05)
    stop["v"] = True
    for t in ts:
        t.join(timeout=2)
    os.close(fd)


def _cum_mb(el: float, ramp_s: float, peak: float) -> float:
    """Integral of _rate() from 0 to el (closed form)."""
    if ramp_s <= 0:
        return peak * el
    if el <= ramp_s:
        return peak * (0.1 * el + 0.45 * el * el / ramp_s)
    return peak * (0.55 * ramp_s + (el - ramp_s))


def w_burst(path: str, seconds: float, mb_per_s: float, ramp_s: float, fsync_end: bool,
            seed: int = 0):
    """Buffered writes: dirty pages accumulate, writeback hits the device later."""
    _set_comm("iobn-burst")
    stop = _install_term()
    chunk = os.urandom(1 << 20)
    t0 = time.monotonic()
    written = 0.0
    with open(path, "wb", buffering=0) as f:
        while not stop["v"] and time.monotonic() - t0 < seconds:
            el = time.monotonic() - t0
            target = _cum_mb(el, ramp_s, mb_per_s)
            if written < target:
                f.write(chunk)
                written += 1
            else:
                time.sleep(0.01)
        if fsync_end:
            os.fsync(f.fileno())


def w_decoy(path: str, seconds: float, seed: int = 0):
    """Re-read a small cached file: large logical reads, ~zero physical reads."""
    _set_comm("iobn-decoy")
    stop = _install_term()
    with open(path, "rb") as f:
        f.read()  # warm the cache once
    t0 = time.monotonic()
    while not stop["v"] and time.monotonic() - t0 < seconds:
        with open(path, "rb") as f:
            while f.read(1 << 20):
                pass


# --------------------------------------------------------------------------
# Scenario planning
# --------------------------------------------------------------------------
@dataclass
class Event:
    t_offset: float
    duration: float
    role: str
    params: Dict = field(default_factory=dict)


def plan(scenario: str, duration_s: float, seed: int) -> List[Event]:
    """Alternating quiet gaps and heavy events, reproducible from the seed."""
    rnd = random.Random(seed)
    mixes = {
        "burst": ["burst"],
        "mixed": ["db"],
        "rogue": ["rogue", "rogue", "decoy+rogue"],
        "all": ["burst", "db", "rogue", "decoy+rogue"],
    }
    if scenario not in mixes:
        raise ValueError(f"unknown scenario {scenario}; choose from {list(mixes)}")
    ev: List[Event] = []
    t = rnd.uniform(8, 14)  # initial quiet period (also lets baselines settle)
    while t < duration_s - 10:
        kind = rnd.choice(mixes[scenario])
        dur = rnd.uniform(6, 14)
        ramp = rnd.choice([0.0, rnd.uniform(2, 6), rnd.uniform(3, 8)])
        if kind == "burst":
            ev.append(Event(t, dur, "burst", {"mb_per_s": rnd.choice([150, 300, 600]),
                                              "ramp_s": ramp, "fsync_end": rnd.random() < 0.5}))
        elif kind == "db":
            ev.append(Event(t, dur, "db", {"workers": rnd.choice([2, 3, 4]), "threads": 3,
                                           "peak_ops": 0, "read_frac": rnd.uniform(0.4, 0.8),
                                           "ramp_s": ramp}))
        else:
            if kind.startswith("decoy"):
                ev.append(Event(max(0, t - 3), dur + 6, "decoy", {}))
            ev.append(Event(t, dur, "rogue", {"peak_ops": rnd.choice([0, 0, 4000]),
                                              "threads": rnd.choice([4, 8, 12]), "ramp_s": ramp}))
        t += dur + rnd.uniform(8, 20)
    return ev


def _start_ticks(pid: int) -> Optional[int]:
    for _ in range(20):
        try:
            return parse_proc_stat(read_file(f"/proc/{pid}/stat"))[3]
        except (OSError, ValueError, IndexError):
            time.sleep(0.005)
    return None


def run_scenario(scenario: str, workdir: str, duration_s: float, gt_path: str, seed: int = 1,
                 data_mb: int = 512, bg_ops: float = 60.0) -> List[Dict]:
    """Execute a planned scenario; returns and writes ground-truth rows."""
    _set_comm("iobn-runner")
    ctx = mp.get_context("fork")
    os.makedirs(workdir, exist_ok=True)
    free_mb = shutil.disk_usage(workdir).free >> 20
    if free_mb < data_mb * 3 + 2048:
        raise RuntimeError(f"need ~{data_mb * 3 + 2048} MB free in {workdir}, have {free_mb}")
    db_file = os.path.join(workdir, "db.dat")
    rogue_file = os.path.join(workdir, "rogue.dat")
    decoy_file = os.path.join(workdir, "decoy.dat")
    prefill(db_file, data_mb)
    prefill(rogue_file, max(64, data_mb // 4))
    prefill(decoy_file, 64)

    events = plan(scenario, duration_s, seed)
    gt: List[Dict] = []
    live: List[tuple] = []
    t0 = time.monotonic()

    def spawn(role, target, args, expected, params, end_at):
        p = ctx.Process(target=target, args=args, daemon=False)
        p.start()
        row = {"role": role, "pid": p.pid, "start_ticks": _start_ticks(p.pid),
               "t_start": time.monotonic(), "t_end": None,
               "expected_contributor": expected, "params": params}
        gt.append(row)
        live.append((p, row, end_at))

    # Background load for the whole run — present in every window, never a culprit.
    spawn("bg", w_random_io, (db_file, duration_s + 5, bg_ops, 0.7, 0.0, (4096,), 16, seed, "iobn-bg"),
          False, {"peak_ops": bg_ops}, t0 + duration_s + 5)

    pending = sorted(events, key=lambda e: e.t_offset)
    k = 0
    try:
        while time.monotonic() - t0 < duration_s or live:
            now = time.monotonic()
            while pending and now - t0 >= pending[0].t_offset:
                e = pending.pop(0)
                k += 1
                end_at = now + e.duration
                if e.role == "burst":
                    path = os.path.join(workdir, f"burst_{k}.dat")
                    spawn("burst", w_burst, (path, e.duration, e.params["mb_per_s"],
                                             e.params["ramp_s"], e.params["fsync_end"], seed + k),
                          True, dict(e.params, path=path), end_at)
                elif e.role == "db":
                    for wi in range(e.params["workers"]):
                        spawn("db", w_random_io, (db_file, e.duration, e.params["peak_ops"],
                                                  e.params["read_frac"], e.params["ramp_s"],
                                                  BS_CHOICES, 8, seed * 100 + k * 10 + wi, "iobn-db",
                                                  e.params["threads"]),
                              True, e.params, end_at)
                elif e.role == "rogue":
                    spawn("rogue", w_random_io, (rogue_file, e.duration, e.params["peak_ops"], 0.5,
                                                 e.params["ramp_s"], (4096,), 1, seed + k, "iobn-rogue",
                                                 e.params["threads"]),
                          True, e.params, end_at)
                elif e.role == "decoy":
                    spawn("decoy", w_decoy, (decoy_file, e.duration, seed + k), False, {}, end_at)
            for item in list(live):
                p, row, end_at = item
                if not p.is_alive() or time.monotonic() >= end_at + 2.0:
                    if p.is_alive():
                        p.terminate()
                    p.join(timeout=5)
                    row["t_end"] = time.monotonic()
                    live.remove(item)
                    if row["role"] == "burst":
                        # Unlinking before writeback finishes turns dirty pages into
                        # cancelled_write_bytes — attribution must handle that too.
                        try:
                            os.unlink(row["params"]["path"])
                        except OSError:
                            pass
            if not pending and time.monotonic() - t0 >= duration_s:
                for p, row, _ in live:
                    p.terminate()
            time.sleep(0.02)
    finally:
        for p, row, _ in live:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
            row["t_end"] = row["t_end"] or time.monotonic()
        for f in os.listdir(workdir):
            if f.startswith("burst_"):
                try:
                    os.unlink(os.path.join(workdir, f))
                except OSError:
                    pass
        with open(gt_path, "w") as fh:
            for r in gt:
                fh.write(json.dumps(r) + "\n")
    return gt
