"""Phase 1 — Sampling loop: tiered, drift-free, self-measuring.

Output (one run directory):
    device.csv   one row per tick per monitored device: raw counter deltas,
                 per-interval device metrics, host pressure (PSI, iowait,
                 steal, Dirty/Writeback), and validity flags.
    process.csv  sparse long table: one row per (process tick, process that
                 did I/O). Process-level THROUGHPUT only — no latency columns.
    meta.json    environment, devices, coverage counters, measured overhead.

Tiering: device/host files are tiny and fixed-size, read every tick
(default 250 ms). The /proc/[pid] scan is O(#processes), so it runs every
`proc_every` ticks (default 4 -> 1 s). This is the honest version of the
paper's scalability claim: device ingestion is O(#devices) per tick,
process ingestion is O(#processes) per process tick — not O(1).
"""
from __future__ import annotations

import csv
import json
import os
import platform
import resource
import signal
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from . import telemetry as T

DEVICE_METRIC_COLS = [
    "in_flight", "r_iops", "w_iops", "r_bps", "w_bps", "r_await_ms", "w_await_ms",
    "util", "aqu_sz",
]
RAW_DELTA_COLS = ["d_" + f for f in T.DISKSTAT_FIELDS if f not in T.GAUGES]
HOST_COLS = ["d_psi_some_us", "d_psi_full_us", "iowait_frac", "steal_frac",
             "mem_dirty_kb", "mem_writeback_kb"]
DEVICE_CSV_COLS = (["ts", "wall", "dt_s", "device", "valid", "gap", "suspend"]
                   + RAW_DELTA_COLS + DEVICE_METRIC_COLS + HOST_COLS)
PROC_CSV_COLS = (["ts", "dt_s", "pid", "start_ticks", "comm", "ppid", "kthread",
                  "born_in_interval", "reaped_child_adjusted", "d_blkio_ticks"]
                 + ["d_" + f for f in T.PROC_IO_FIELDS])


@dataclass
class CollectorConfig:
    devices: Optional[List[str]] = None   # None -> all whole disks
    interval_s: float = 0.25
    proc_every: int = 4                   # process scan every N ticks
    gap_factor: float = 2.0               # dt > gap_factor*interval -> gap flag
    suspend_threshold_s: float = 0.5      # boottime-monotonic jump -> suspend flag
    flush_every: int = 20


@dataclass
class CoverageStats:
    ticks: int = 0
    invalid_ticks: int = 0
    gaps: int = 0
    suspends: int = 0
    overruns: int = 0                     # tick work took longer than the interval
    proc_scans: int = 0
    procs_seen: int = 0
    procs_vanished: int = 0
    procs_io_denied: int = 0
    procs_malformed: int = 0
    max_tick_work_ms: float = 0.0
    sum_tick_work_ms: float = 0.0


class Collector:
    """Usable two ways: `run(duration)` to write CSVs, or `step()` driven by a
    caller (the live predictor) via `on_tick` callbacks."""

    def __init__(self, cfg: CollectorConfig, out_dir: Optional[str] = None,
                 on_tick: Optional[Callable[[float, List[Dict], List[Dict]], None]] = None):
        self.cfg = cfg
        self.devices = cfg.devices or T.list_whole_disks()
        if not self.devices:
            raise RuntimeError("no block devices found in /sys/block")
        self.out_dir = out_dir
        self.on_tick = on_tick
        self.cov = CoverageStats()
        self.tracker = T.ProcessTracker()
        self._stop = False
        self._prev_disk = None
        self._prev_clock = None
        self._prev_psi = None
        self._prev_cpu = None
        self._last_proc_clock = None
        self._dev_f = self._proc_f = None
        self._dev_w = self._proc_w = None

    # ---------------------------------------------------------------- io
    def _open_outputs(self):
        if not self.out_dir:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        self._dev_f = open(os.path.join(self.out_dir, "device.csv"), "w", newline="")
        self._proc_f = open(os.path.join(self.out_dir, "process.csv"), "w", newline="")
        self._dev_w = csv.DictWriter(self._dev_f, DEVICE_CSV_COLS, extrasaction="ignore")
        self._proc_w = csv.DictWriter(self._proc_f, PROC_CSV_COLS, extrasaction="ignore")
        self._dev_w.writeheader()
        self._proc_w.writeheader()

    def _close_outputs(self):
        for f in (self._dev_f, self._proc_f):
            if f:
                f.close()

    # -------------------------------------------------------------- sample
    def _prime(self):
        self._prev_clock = T.clock_snapshot()
        self._prev_disk = T.read_diskstats(self.devices)
        missing = set(self.devices) - set(self._prev_disk)
        if missing:
            raise RuntimeError(f"devices not in /proc/diskstats: {sorted(missing)}")
        self._prev_psi = T.read_psi_io()
        self._prev_cpu = T.read_cpu_times()
        procs, _ = T.scan_processes()
        self.tracker.update(procs, self._prev_clock["mono"], T.uptime_s())
        self._last_proc_clock = self._prev_clock

    def step(self, tick_index: int):
        clk = T.clock_snapshot()
        disk = T.read_diskstats(self.devices)
        psi = T.read_psi_io()
        cpu = T.read_cpu_times()
        mem = T.read_meminfo_writeback() or {}

        dt = clk["mono"] - self._prev_clock["mono"]
        boot_jump = (clk["boot"] - self._prev_clock["boot"]) - dt
        gap = dt > self.cfg.gap_factor * self.cfg.interval_s
        suspend = boot_jump > self.cfg.suspend_threshold_s

        host = {"mem_dirty_kb": mem.get("mem_dirty_kb", ""),
                "mem_writeback_kb": mem.get("mem_writeback_kb", "")}
        if psi and self._prev_psi:
            host["d_psi_some_us"] = psi.get("psi_some_total_us", 0) - self._prev_psi.get("psi_some_total_us", 0)
            host["d_psi_full_us"] = psi.get("psi_full_total_us", 0) - self._prev_psi.get("psi_full_total_us", 0)
        if cpu and self._prev_cpu:
            tot = cpu["cpu_total"] - self._prev_cpu["cpu_total"]
            if tot > 0:
                # iowait is a weak, per-CPU-idle artefact (a CPU is "iowait" only if
                # idle while a task it last ran waits on I/O). Context feature only.
                host["iowait_frac"] = (cpu["cpu_iowait"] - self._prev_cpu["cpu_iowait"]) / tot
                # steal > 0 means the hypervisor ran something else on our vCPU
                # (WSL2/VMs) — it distorts every time-based metric in the tick.
                host["steal_frac"] = (cpu["cpu_steal"] - self._prev_cpu["cpu_steal"]) / tot

        dev_rows: List[Dict] = []
        any_invalid = False
        for name in self.devices:
            m = None
            if name in disk and name in self._prev_disk:
                m = T.device_interval(self._prev_disk[name], disk[name], dt)
            valid = m is not None and not gap and not suspend
            any_invalid |= not valid
            row = {"ts": round(clk["mono"], 4), "wall": round(clk["wall"], 3),
                   "dt_s": round(dt, 5), "device": name, "valid": int(valid),
                   "gap": int(gap), "suspend": int(suspend)}
            if m:
                row.update(m)
            row.update(host)
            dev_rows.append(row)

        proc_rows: List[Dict] = []
        if tick_index % self.cfg.proc_every == 0:
            procs, st = T.scan_processes()
            pdt = clk["mono"] - self._last_proc_clock["mono"]
            for r in self.tracker.update(procs, clk["mono"], T.uptime_s()):
                r["ts"], r["dt_s"] = round(clk["mono"], 4), round(pdt, 5)
                proc_rows.append(r)
            self._last_proc_clock = clk
            c = self.cov
            c.proc_scans += 1
            c.procs_seen += st.seen
            c.procs_vanished += st.vanished
            c.procs_io_denied += st.io_denied
            c.procs_malformed += st.malformed

        self._prev_clock, self._prev_disk = clk, disk
        self._prev_psi, self._prev_cpu = psi, cpu
        self.cov.ticks += 1
        self.cov.invalid_ticks += int(any_invalid)
        self.cov.gaps += int(gap)
        self.cov.suspends += int(suspend)
        return dev_rows, proc_rows

    # ----------------------------------------------------------------- run
    def stop(self, *_):
        self._stop = True

    def run(self, duration_s: Optional[float] = None) -> Dict:
        self._open_outputs()
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        self._prime()
        t0 = time.monotonic()
        k = 0
        try:
            while not self._stop:
                k += 1
                # Absolute deadlines: no cumulative drift from work time.
                deadline = t0 + k * self.cfg.interval_s
                sleep = deadline - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    self.cov.overruns += 1
                w0 = time.perf_counter()
                dev_rows, proc_rows = self.step(k)
                if self._dev_w:
                    self._dev_w.writerows(dev_rows)
                    self._proc_w.writerows(proc_rows)
                    if k % self.cfg.flush_every == 0:
                        self._dev_f.flush()
                        self._proc_f.flush()
                if self.on_tick:
                    self.on_tick(dev_rows[0]["ts"], dev_rows, proc_rows)
                work_ms = (time.perf_counter() - w0) * 1000
                self.cov.max_tick_work_ms = max(self.cov.max_tick_work_ms, work_ms)
                self.cov.sum_tick_work_ms += work_ms
                if duration_s is not None and time.monotonic() - t0 >= duration_s:
                    break
        finally:
            elapsed = time.monotonic() - t0
            ru1 = resource.getrusage(resource.RUSAGE_SELF)
            cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
            meta = self._meta(elapsed, cpu_s)
            if self.out_dir:
                with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
            self._close_outputs()
        return meta

    def _meta(self, elapsed: float, cpu_s: float) -> Dict:
        ncpu = os.cpu_count() or 1
        cov = asdict(self.cov)
        cov["mean_tick_work_ms"] = cov.pop("sum_tick_work_ms") / max(1, self.cov.ticks)
        denied_frac = self.cov.procs_io_denied / max(1, self.cov.procs_seen)
        return {
            "kernel": platform.release(),
            "devices": self.devices,
            "config": asdict(self.cfg),
            "elapsed_s": round(elapsed, 2),
            "psi_available": T.read_psi_io() is not None,
            "euid": os.geteuid(),
            "coverage": cov,
            "proc_io_denied_frac": round(denied_frac, 4),
            # Measured, not asserted: collector CPU as % of one core and of all cores.
            "overhead": {
                "cpu_seconds": round(cpu_s, 3),
                "pct_of_one_core": round(100 * cpu_s / max(elapsed, 1e-9), 3),
                "pct_of_total_capacity": round(100 * cpu_s / max(elapsed, 1e-9) / ncpu, 3),
                "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "ncpu": ncpu,
            },
            "warnings": ([f"{denied_frac:.0%} of processes had unreadable /proc/pid/io; "
                          "run as root or grant CAP_SYS_PTRACE for full attribution"]
                         if denied_frac > 0.05 else []),
        }
