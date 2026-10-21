"""Phase 5 — WSL2 telemetry validation.

What is different under WSL2
----------------------------
* The Linux distro runs in a Hyper-V utility VM. Its root filesystem is an
  ext4 image (ext4.vhdx) exposed as a virtual SCSI disk (usually /dev/sdc or
  /dev/sdd; check with device_for_path("/")). diskstats measures the GUEST
  block layer only: the Windows host page cache, the vhdx layer and the
  physical SSD are invisible. Host caching typically makes guest latencies
  look unrealistically low, punctuated by large spikes when the host flushes.
* /mnt/c, /mnt/d ... are Windows drives over 9p/drvfs (or virtiofs). Their
  I/O never enters the guest block layer and is ABSENT from /proc/diskstats.
  Workloads must run on the ext4 side (e.g. ~/iobn-work), never /mnt/c.
* WSL1 has no real block devices; telemetry from WSL1 is meaningless.
* vCPUs can be descheduled by the host (steal time), the VM can be paused
  when Windows sleeps (CLOCK_BOOTTIME jumps vs CLOCK_MONOTONIC), and the
  guest wall clock is periodically re-synced from the host (wall time steps).
  The collector already flags gaps and suspends; this module quantifies them.
* PSI and delay accounting depend on the WSL kernel config; detect, don't assume.

Validation steps (run `iobn wsl-check`):
  1. environment()            what kind of kernel / disks / mounts we have
  2. sector_accounting_check  write+read N MB with O_DIRECT; diskstats must see it
  3. iostat_crosscheck        our counters vs sysstat's iostat over the same period
  4. timing_anomaly_scan      on any recorded run: gaps, suspends, steal, ms quantisation,
                              util > 100%, latency bimodality
Absolute latency numbers from WSL2 describe the virtual disk, not the SSD:
train per storage profile (Constraint in the paper) and reproduce headline
results on native Linux before publishing them.
"""
from __future__ import annotations

import json
import mmap
import os
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import telemetry as T

NON_BLOCK_FS = {"9p", "drvfs", "virtiofs", "fuse", "fuseblk", "tmpfs", "overlay", "nfs", "nfs4",
                "cifs", "smb3"}


def _mounts() -> List[Dict]:
    out = []
    try:
        for line in T.read_file("/proc/self/mountinfo").splitlines():
            left, _, right = line.partition(" - ")
            l, r = left.split(), right.split()
            out.append({"mount_point": l[4], "majmin": l[2], "fstype": r[0],
                        "source": r[1] if len(r) > 1 else ""})
    except OSError:
        pass
    return out


def environment() -> Dict:
    rel = platform.release()
    try:
        version = T.read_file("/proc/version").strip()
    except OSError:
        version = ""
    is_wsl = "microsoft" in (rel + version).lower() or "WSL_DISTRO_NAME" in os.environ
    wsl_version = None
    if is_wsl:
        # WSL2 kernels are "*-microsoft-standard-WSL2" and expose virtual SCSI
        # disks; WSL1 reports "*-Microsoft" and has no guest block devices.
        has_disks = any(n.startswith(("sd", "vd", "nvme")) for n in T.list_whole_disks())
        wsl_version = 2 if ("wsl2" in rel.lower() or has_disks) else 1
    mounts = _mounts()
    windows_mounts = [m for m in mounts if m["mount_point"].startswith("/mnt/") and
                      m["fstype"] in NON_BLOCK_FS]
    try:
        delayacct = T.read_file("/proc/sys/kernel/task_delayacct").strip()
    except OSError:
        delayacct = "unavailable"
    return {
        "kernel": rel,
        "is_wsl": is_wsl,
        "wsl_version": wsl_version,
        "distro": os.environ.get("WSL_DISTRO_NAME"),
        "block_devices": T.list_whole_disks(),
        "root_device": T.device_for_path("/"),
        "home_device": T.device_for_path(os.path.expanduser("~")),
        "windows_mounts_invisible_to_diskstats": [(m["mount_point"], m["fstype"]) for m in windows_mounts],
        "psi_io_available": T.read_psi_io() is not None,
        "delayacct": delayacct,   # "1" -> delayacct_blkio_ticks populated (victim signal)
        "euid": os.geteuid(),
        "clk_tck": T.CLK_TCK,
        "iostat_available": shutil.which("iostat") is not None,
        "ncpu": os.cpu_count(),
    }


def check_workdir(path: str) -> Dict:
    """FAIL if workload I/O in `path` would be invisible to diskstats."""
    dev = T.device_for_path(path)
    fstype = next((m["fstype"] for m in sorted(_mounts(), key=lambda m: -len(m["mount_point"]))
                   if os.path.realpath(path).startswith(m["mount_point"])), "?")
    ok = dev is not None
    return {"path": path, "device": dev, "fstype": fstype, "status": "PASS" if ok else "FAIL",
            "detail": ("I/O lands on a guest block device" if ok else
                       f"{fstype} is not backed by a guest block device; its I/O will not appear "
                       "in /proc/diskstats (on WSL2 use the ext4 side, e.g. ~/iobn-work, not /mnt/c)")}


# --------------------------------------------------------------------------
# 2. Sector accounting: does diskstats see what we know we did?
# --------------------------------------------------------------------------
def sector_accounting_check(path: str, mb: int = 128) -> Dict:
    dev = T.device_for_path(path)
    if dev is None:
        return {"status": "FAIL", "detail": "path not on a block device"}
    fn = os.path.join(path, f".iobn_acct_{os.getpid()}.dat")
    size = mb << 20
    buf = mmap.mmap(-1, 1 << 20)
    buf.write(os.urandom(1 << 20))
    res: Dict = {"device": dev, "mb": mb}
    direct = True
    try:
        try:
            fd = os.open(fn, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)
        except OSError:
            fd = os.open(fn, os.O_CREAT | os.O_RDWR, 0o600)
            direct = False
        res["o_direct"] = direct
        s0 = T.read_diskstats([dev])[dev]
        t0 = time.monotonic()
        for i in range(mb):
            os.pwritev(fd, [buf], i << 20)
        os.fsync(fd)
        t1 = time.monotonic()
        s1 = T.read_diskstats([dev])[dev]
        if not direct:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        for i in range(mb):
            os.preadv(fd, [buf], i << 20)
        t2 = time.monotonic()
        s2 = T.read_diskstats([dev])[dev]
        os.close(fd)
    finally:
        try:
            os.unlink(fn)
        except OSError:
            pass
    I = T.IDX
    w_bytes = (s1[I["wr_sectors"]] - s0[I["wr_sectors"]]) * T.SECTOR_BYTES
    r_bytes = (s2[I["rd_sectors"]] - s1[I["rd_sectors"]]) * T.SECTOR_BYTES
    w_ratio, r_ratio = w_bytes / size, r_bytes / size
    # Writes: data + ext4 journal/metadata, plus any concurrent system writes.
    w_ok = 0.95 <= w_ratio <= 1.6
    r_ok = 0.95 <= r_ratio <= 1.3
    res.update({
        "write_ratio": round(w_ratio, 3), "read_ratio": round(r_ratio, 3),
        "write_mb_s": round(mb / max(t1 - t0, 1e-6), 1), "read_mb_s": round(mb / max(t2 - t1, 1e-6), 1),
        "w_await_ms": round((s1[I["wr_ticks"]] - s0[I["wr_ticks"]]) / max(1, s1[I["wr_ios"]] - s0[I["wr_ios"]]), 3),
        "r_await_ms": round((s2[I["rd_ticks"]] - s1[I["rd_ticks"]]) / max(1, s2[I["rd_ios"]] - s1[I["rd_ios"]]), 3),
        "status": "PASS" if (w_ok and r_ok) else ("WARN" if (w_ratio > 0.5 and r_ratio > 0.5) else "FAIL"),
    })
    if r_ratio < 0.5:
        res["detail"] = ("reads did not reach the guest block device — page cache or a non-block "
                         "filesystem is serving them; latency features would be blind")
    elif w_ratio > 1.6:
        res["detail"] = "write amplification or concurrent background writes; rerun on an idle system"
    return res


# --------------------------------------------------------------------------
# 3. iostat cross-check
# --------------------------------------------------------------------------
def _iostat_disk_rows(js: Dict, dev: str) -> List[Dict]:
    rows = []
    for host in js.get("sysstat", {}).get("hosts", []):
        for st in host.get("statistics", []):
            for d in st.get("disk", []):
                if d.get("disk_device") == dev:
                    rows.append(d)
    return rows


def _g(d: Dict, *names, default=np.nan):
    for n in names:  # field names changed across sysstat versions
        if n in d:
            return float(d[n])
    return default


def iostat_crosscheck(dev: str, seconds: int = 15, load_path: Optional[str] = None) -> Dict:
    """Run `iostat -dxy -o JSON dev 1 N` alongside our own 1 s sampler and compare
    run-level aggregates (sampling windows are not phase-aligned, so per-second
    rows legitimately differ; totals and I/O-weighted latency must agree)."""
    if shutil.which("iostat") is None:
        return {"status": "SKIP", "detail": "install sysstat (sudo apt install sysstat)"}
    loader = None
    if load_path:
        from .workloads import prefill, w_random_io
        import multiprocessing as mp
        f = os.path.join(load_path, ".iobn_xcheck.dat")
        prefill(f, 128)
        loader = mp.get_context("fork").Process(
            target=w_random_io, args=(f, seconds + 3, 400.0, 0.6, 0.0, (4096, 65536), 8, 3, "iobn-xcheck"))
        loader.start()
        time.sleep(1.0)
    proc = subprocess.Popen(["iostat", "-dxy", "-o", "JSON", dev, "1", str(seconds)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=dict(os.environ, LC_ALL="C", S_TIME_FORMAT="ISO"))
    s0 = T.read_diskstats([dev])[dev]
    t0 = time.monotonic()
    out, err = proc.communicate(timeout=seconds + 20)
    t1 = time.monotonic()
    s1 = T.read_diskstats([dev])[dev]
    if loader:
        loader.terminate()
        loader.join()
        try:
            os.unlink(os.path.join(load_path, ".iobn_xcheck.dat"))
        except OSError:
            pass
    try:
        rows = _iostat_disk_rows(json.loads(out), dev)
    except json.JSONDecodeError:
        return {"status": "FAIL", "detail": f"could not parse iostat JSON: {err.strip()[:200]}"}
    if not rows:
        return {"status": "FAIL", "detail": f"iostat reported no rows for {dev}"}

    I, el = T.IDX, t1 - t0
    ours = {
        "r_ios": s1[I["rd_ios"]] - s0[I["rd_ios"]], "w_ios": s1[I["wr_ios"]] - s0[I["wr_ios"]],
        "r_kb": (s1[I["rd_sectors"]] - s0[I["rd_sectors"]]) / 2.0,
        "w_kb": (s1[I["wr_sectors"]] - s0[I["wr_sectors"]]) / 2.0,
    }
    ours["await_ms"] = ((s1[I["rd_ticks"]] - s0[I["rd_ticks"]]) + (s1[I["wr_ticks"]] - s0[I["wr_ticks"]])) \
        / max(1, ours["r_ios"] + ours["w_ios"])
    ours["util_pct"] = 100 * (s1[I["io_ticks"]] - s0[I["io_ticks"]]) / (el * 1000)
    rs = np.array([_g(r, "r/s") for r in rows]); ws = np.array([_g(r, "w/s") for r in rows])
    ra = np.array([_g(r, "r_await") for r in rows]); wa = np.array([_g(r, "w_await") for r in rows])
    theirs = {
        "r_ios": float(rs.sum()), "w_ios": float(ws.sum()),
        "r_kb": float(sum(_g(r, "rkB/s") for r in rows)), "w_kb": float(sum(_g(r, "wkB/s") for r in rows)),
        "await_ms": float((np.nansum(rs * np.nan_to_num(ra)) + np.nansum(ws * np.nan_to_num(wa)))
                          / max(1.0, rs.sum() + ws.sum())),
        "util_pct": float(np.nanmean([_g(r, "util", "%util") for r in rows])),
    }
    # Scale our totals to iostat's N one-second intervals.
    scale = len(rows) / el
    cmp = {}
    worst = 0.0
    for k in ("r_ios", "w_ios", "r_kb", "w_kb", "await_ms", "util_pct"):
        o = ours[k] * (scale if k in ("r_ios", "w_ios", "r_kb", "w_kb") else 1.0)
        t = theirs[k]
        rel = abs(o - t) / max(abs(t), 1e-9) if max(abs(o), abs(t)) > 1e-6 else 0.0
        # tiny absolute values (idle) are not meaningful relative errors
        if max(abs(o), abs(t)) < (5 if k != "await_ms" else 0.05):
            rel = 0.0
        cmp[k] = {"ours": round(o, 3), "iostat": round(t, 3), "rel_err": round(rel, 3)}
        worst = max(worst, rel)
    status = "PASS" if worst <= 0.10 else ("WARN" if worst <= 0.25 else "FAIL")
    return {"status": status, "device": dev, "seconds": len(rows), "comparison": cmp,
            "worst_rel_err": round(worst, 3),
            "note": "iostat derives the same fields from /proc/diskstats; disagreement means "
                    "a parsing/delta bug or sampling-window skew, not a different truth"}


# --------------------------------------------------------------------------
# 4. Timing anomaly scan on a recorded run
# --------------------------------------------------------------------------
def timing_anomaly_scan(device_csv: str, device: Optional[str] = None,
                        interval_s: Optional[float] = None) -> Dict:
    d = pd.read_csv(device_csv)
    if device:
        d = d[d["device"] == device]
    else:
        device = d["device"].iloc[0]
        d = d[d["device"] == device]
    dt = d["dt_s"].to_numpy()
    interval_s = interval_s or float(np.median(dt))
    n = len(d)
    ios = d["d_rd_ios"] + d["d_wr_ios"]
    ticks = d["d_rd_ticks"] + d["d_wr_ticks"]
    busy = ios > 0
    zero_lat = float(((ticks == 0) & busy).sum() / max(1, busy.sum()))
    per_tick_await = (ticks / ios).where(busy)
    p50 = float(np.nanmedian(per_tick_await)) if busy.any() else float("nan")
    p99 = float(np.nanpercentile(per_tick_await, 99)) if busy.any() else float("nan")
    wall_jump = np.abs(np.diff(d["wall"].to_numpy()) - np.diff(d["ts"].to_numpy()))
    findings = {
        "device": device, "rows": n, "interval_s": round(interval_s, 4),
        "dt_jitter_ms_std": round(float(np.std(dt) * 1000), 3),
        "gap_rows": int(d["gap"].sum()), "suspend_rows": int(d["suspend"].sum()),
        "invalid_frac": round(float(1 - d["valid"].mean()), 4),
        "util_over_100pct_rows": int((d["util"] > 1.05).sum()),
        "zero_latency_completion_frac": round(zero_lat, 3),
        "await_p50_ms": round(p50, 3), "await_p99_ms": round(p99, 3),
        "await_p99_over_p50": round(p99 / p50, 1) if p50 and p50 > 0 else None,
        "steal_mean": round(float(d["steal_frac"].mean()), 4) if "steal_frac" in d else None,
        "steal_max": round(float(d["steal_frac"].max()), 4) if "steal_frac" in d else None,
        "wall_clock_steps": int((wall_jump > 0.25).sum()),
    }
    warn = []
    if findings["gap_rows"] / max(1, n) > 0.01:
        warn.append("collector gaps > 1% of ticks: host is descheduling the VM or the collector overruns; "
                    "these rows are excluded from windows and labels")
    if findings["suspend_rows"]:
        warn.append("VM suspend/resume detected (BOOTTIME jump): label windows crossing it were dropped")
    if findings["util_over_100pct_rows"]:
        warn.append("util > 100%: io_ticks advanced faster than our clock (virtual-time skew); "
                    "treat util as advisory on this platform")
    if zero_lat > 0.3:
        warn.append(f"{zero_lat:.0%} of busy ticks report 0 ms total latency: completions are faster "
                    "than diskstats' 1 ms resolution (host cache). Use window-aggregated await, not per-tick")
    if findings["await_p99_over_p50"] and findings["await_p99_over_p50"] > 50:
        warn.append("extreme latency bimodality (p99/p50 > 50): consistent with host-side cache flushes; "
                    "use per-profile baselines and do not compare absolute ms with bare metal")
    if findings["steal_max"] and findings["steal_max"] > 0.05:
        warn.append("CPU steal > 5% observed: host contention distorts time-based metrics in those ticks")
    if findings["wall_clock_steps"]:
        warn.append("wall clock stepped (host time sync); all features use CLOCK_MONOTONIC, unaffected")
    findings["warnings"] = warn
    findings["status"] = "PASS" if not warn else "WARN"
    return findings


def run_wsl_validation(workdir: str, iostat_seconds: int = 15) -> Dict:
    env = environment()
    report = {"environment": env, "workdir": check_workdir(workdir)}
    if env["is_wsl"] and env["wsl_version"] == 1:
        report["fatal"] = "WSL1 has no guest block layer; use WSL2 (wsl --set-version <distro> 2)"
        return report
    if report["workdir"]["status"] == "FAIL":
        return report
    report["sector_accounting"] = sector_accounting_check(workdir)
    dev = report["workdir"]["device"]
    report["iostat_crosscheck"] = iostat_crosscheck(dev, iostat_seconds, load_path=workdir)
    return report
