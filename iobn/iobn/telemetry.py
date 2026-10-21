"""Phase 1 — Telemetry ingestion and parsing.

Design rules (fixes from the architecture review)
-------------------------------------------------
1. Two metric families, never mixed:

   * DEVICE-LEVEL latency/pressure, from /proc/diskstats (+ PSI, /proc/stat,
     /proc/meminfo). diskstats gives *aggregate* per-device time spent on
     completed requests (rd_ticks / wr_ticks) and the integral of in-flight
     requests (time_in_queue). Dividing ticks by completed I/Os gives an
     average request latency for the *device*, measured in the block layer
     (queueing + service). It says nothing about which process waited.

   * PROCESS-LEVEL throughput, from /proc/[pid]/io. These are byte and syscall
     counters charged to the task. There is no per-process queue wait time in
     either file. (The only per-process *delay* counter is
     delayacct_blkio_ticks in /proc/[pid]/stat, which needs delay accounting
     enabled, and it measures how long a task was *blocked* — a victim signal,
     not a cause signal. We collect it optionally and keep it separate.)

2. Short-lived processes are normal, not exceptional: a PID can vanish between
   listdir() and open(), between open() and read(), or be reused by a new
   process. Every per-PID read is wrapped, and processes are keyed by
   (pid, starttime) so PID reuse can never create a bogus negative/positive
   delta.

3. Overhead: raw os.open/os.read (no psutil, no subprocesses), one read per
   file, integers parsed directly. Device files are read every tick; the
   O(#processes) scan can run on a slower tier (see collector.py).
"""
from __future__ import annotations

import errno
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# /proc/diskstats always counts 512-byte sectors regardless of the device's
# logical block size (Documentation/admin-guide/iostats.rst).
SECTOR_BYTES = 512
CLK_TCK = os.sysconf("SC_CLK_TCK")
PF_KTHREAD = 0x00200000  # task flag for kernel threads (include/linux/sched.h)

# Field order after "major minor name". 11 fields on <4.18, 15 on 4.18+
# (discards), 17 on 5.5+ (flushes). Missing trailing fields are filled with 0.
DISKSTAT_FIELDS: Tuple[str, ...] = (
    "rd_ios", "rd_merges", "rd_sectors", "rd_ticks",
    "wr_ios", "wr_merges", "wr_sectors", "wr_ticks",
    "in_flight", "io_ticks", "time_in_queue",
    "dc_ios", "dc_merges", "dc_sectors", "dc_ticks",
    "fl_ios", "fl_ticks",
)
N_DISKSTAT = len(DISKSTAT_FIELDS)
IDX = {name: i for i, name in enumerate(DISKSTAT_FIELDS)}

# in_flight is a gauge (instantaneous), everything else is a cumulative counter.
GAUGES = frozenset({"in_flight"})
# The kernel prints the millisecond fields with "%u" (unsigned int) even on
# 64-bit kernels, so they wrap at 2**32 ms (~49.7 days of accumulated time).
# Busy devices with many parallel requests accumulate rd/wr_ticks faster than
# wall-clock time, so this wrap is reachable in long-running collectors.
MS_FIELDS = frozenset({"rd_ticks", "wr_ticks", "io_ticks", "time_in_queue",
                       "dc_ticks", "fl_ticks"})
U32 = 1 << 32

PROC_IO_FIELDS: Tuple[str, ...] = (
    "rchar", "wchar", "syscr", "syscw",
    "read_bytes", "write_bytes", "cancelled_write_bytes",
)


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def read_file(path: str) -> str:
    """Read a procfs/sysfs file with a single open and as few read()s as possible.

    procfs files report st_size 0, so we loop until EOF. Raises OSError
    subclasses unchanged — callers decide which are expected.
    """
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        chunks = []
        while True:
            b = os.read(fd, 65536)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks).decode("utf-8", "replace")
    finally:
        os.close(fd)


def counter_delta(cur: int, prev: int, field_name: str = "") -> Optional[int]:
    """Delta of a monotonically increasing kernel counter.

    Returns None when the delta cannot be trusted (counter reset because a
    device was removed/re-added, or a stats file was swapped). Callers must
    mark the sample invalid rather than substitute 0.
    """
    d = cur - prev
    if d >= 0:
        return d
    if field_name in MS_FIELDS and prev < U32 and cur < U32:
        return d + U32  # 32-bit millisecond wrap
    return None


def clock_snapshot() -> Dict[str, float]:
    """Monotonic + boottime + wall clocks taken back-to-back.

    CLOCK_MONOTONIC stops during system suspend / VM pause; CLOCK_BOOTTIME
    keeps counting. A jump in (boottime - monotonic) between samples means the
    machine (or the WSL2 utility VM) was suspended: the interval is invalid.
    """
    return {
        "mono": time.clock_gettime(time.CLOCK_MONOTONIC),
        "boot": time.clock_gettime(time.CLOCK_BOOTTIME),
        "wall": time.time(),
    }


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
_SKIP_PREFIXES = ("loop", "ram", "zram", "fd", "sr", "nbd")


def list_whole_disks(include_virtual: bool = False) -> List[str]:
    """Whole-disk names from /sys/block (partitions are excluded so bytes are
    not double counted; loop/zram/ram excluded unless asked)."""
    try:
        names = sorted(os.listdir("/sys/block"))
    except OSError:
        return []
    return [n for n in names if include_virtual or not n.startswith(_SKIP_PREFIXES)]


def device_for_path(path: str) -> Optional[str]:
    """Map a filesystem path to the whole-disk name that serves its I/O.

    Returns None when the filesystem is not backed by a local block device
    (st_dev major 0: 9p / drvfs / virtiofs / tmpfs / overlay / fuse). On WSL2
    this is exactly the /mnt/c case — that I/O is invisible to diskstats.
    Device-mapper / md stacks resolve to the dm-N / mdN node itself, which is
    what the filesystem actually waits on.
    """
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    if major == 0:
        return None
    sys_path = f"/sys/dev/block/{major}:{minor}"
    try:
        real = os.path.realpath(sys_path)
    except OSError:
        return None
    if os.path.exists(os.path.join(real, "partition")):
        real = os.path.dirname(real)  # partition -> parent disk
    name = os.path.basename(real)
    return name if os.path.isdir(f"/sys/block/{name}") else None


# --------------------------------------------------------------------------
# /proc/diskstats
# --------------------------------------------------------------------------
def parse_diskstats(text: str, devices: Optional[Iterable[str]] = None
                    ) -> Dict[str, Tuple[int, ...]]:
    """Parse /proc/diskstats into {device: counters tuple in DISKSTAT_FIELDS order}.

    Tolerates 11/15/17-field kernels and skips malformed lines instead of
    raising, because one odd line must not kill a long-running collector.
    """
    want = set(devices) if devices is not None else None
    out: Dict[str, Tuple[int, ...]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 + 11:
            continue
        name = parts[2]
        if want is not None and name not in want:
            continue
        try:
            vals = [int(x) for x in parts[3:3 + N_DISKSTAT]]
        except ValueError:
            continue
        if len(vals) < N_DISKSTAT:
            vals.extend([0] * (N_DISKSTAT - len(vals)))
        out[name] = tuple(vals)
    return out


def read_diskstats(devices: Optional[Iterable[str]] = None) -> Dict[str, Tuple[int, ...]]:
    return parse_diskstats(read_file("/proc/diskstats"), devices)


def device_interval(prev: Tuple[int, ...], cur: Tuple[int, ...], dt_s: float
                    ) -> Optional[Dict[str, float]]:
    """Raw deltas + per-interval DEVICE metrics for one device.

    Raw deltas (d_*) are kept so Phase 2 can compute I/O-weighted latency over
    any window as sum(d_ticks)/sum(d_ios) instead of averaging per-tick
    averages (which over-weights near-idle ticks and is wrong).

    Returns None if any counter went backwards (reset) — the sample is unusable.
    """
    if dt_s <= 0:
        return None
    d: Dict[str, float] = {}
    for name in DISKSTAT_FIELDS:
        i = IDX[name]
        if name in GAUGES:
            continue
        v = counter_delta(cur[i], prev[i], name)
        if v is None:
            return None
        d["d_" + name] = float(v)

    dt_ms = dt_s * 1000.0
    rd, wr = d["d_rd_ios"], d["d_wr_ios"]
    nan = float("nan")
    m = dict(d)
    m.update({
        "in_flight": float(cur[IDX["in_flight"]]),             # gauge, instantaneous
        "r_iops": rd / dt_s,
        "w_iops": wr / dt_s,
        "r_bps": d["d_rd_sectors"] * SECTOR_BYTES / dt_s,
        "w_bps": d["d_wr_sectors"] * SECTOR_BYTES / dt_s,
        # Device-level average latency per completed request (ms). Block-layer
        # time from request start to completion: queueing + device service.
        # ms resolution: on fast/virtual disks single-tick values quantise to 0
        # — Phase 2 aggregates over windows for exactly this reason.
        "r_await_ms": d["d_rd_ticks"] / rd if rd else nan,
        "w_await_ms": d["d_wr_ticks"] / wr if wr else nan,
        # Fraction of wall time with >=1 request in flight. NOT a saturation
        # measure on NVMe/SSD (many parallel queues): 100% busy can still have
        # spare capacity. Kept as a feature, never used alone as a label.
        "util": d["d_io_ticks"] / dt_ms,
        # Average number of requests in flight (Little's law: aqu = IOPS x await).
        "aqu_sz": d["d_time_in_queue"] / dt_ms,
    })
    return m


# --------------------------------------------------------------------------
# Host-level pressure: PSI, /proc/stat, /proc/meminfo
# --------------------------------------------------------------------------
def read_psi_io() -> Optional[Dict[str, int]]:
    """/proc/pressure/io cumulative stall totals in microseconds.

    'some' = share of time at least one runnable task was stalled on I/O;
    'full' = share of time all non-idle tasks were stalled. We use the
    cumulative 'total' (not the kernel's smoothed avg10) and difference it per
    interval so the signal is not pre-smoothed with a 10-second lag.
    Returns None when PSI is not compiled in or disabled with psi=0
    (read() then fails with EOPNOTSUPP).
    """
    try:
        text = read_file("/proc/pressure/io")
    except OSError:
        return None
    out: Dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        for kv in parts[1:]:
            if kv.startswith("total="):
                try:
                    out[f"psi_{parts[0]}_total_us"] = int(kv[6:])
                except ValueError:
                    pass
    return out or None


def read_cpu_times() -> Optional[Dict[str, int]]:
    """Aggregate CPU jiffies from the first line of /proc/stat."""
    try:
        first = read_file("/proc/stat").split("\n", 1)[0].split()
    except OSError:
        return None
    if not first or first[0] != "cpu":
        return None
    vals = [int(x) for x in first[1:]] + [0] * 10
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    out = {f"cpu_{n}": vals[i] for i, n in enumerate(names)}
    # guest/guest_nice are already included in user/nice — exclude from total.
    out["cpu_total"] = sum(vals[:8])
    return out


def read_meminfo_writeback() -> Optional[Dict[str, int]]:
    """Dirty and Writeback page-cache volume (kB, gauges).

    Buffered writes land in the page cache first and reach the device later
    via writeback, so a rising Dirty curve is one of the few genuinely
    *leading* indicators of an upcoming write-latency spike.
    """
    try:
        text = read_file("/proc/meminfo")
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        if line.startswith(("Dirty:", "Writeback:")):
            k, v = line.split(":", 1)
            out[f"mem_{k.lower()}_kb"] = int(v.split()[0])
    return out or None


# --------------------------------------------------------------------------
# Per-process: /proc/[pid]/stat and /proc/[pid]/io
# --------------------------------------------------------------------------
# Errors that simply mean "this process is gone or not ours" — expected under
# churn. ESRCH shows up when the task exits between open() and read().
_VANISHED = (FileNotFoundError, ProcessLookupError)
_TRANSIENT_ERRNOS = {errno.ESRCH, errno.ENOENT}


@dataclass(frozen=True)
class ProcKey:
    """Identity of a process that survives PID reuse."""
    pid: int
    start_ticks: int  # field 22 of /proc/[pid]/stat, clock ticks since boot


@dataclass
class ProcSample:
    key: ProcKey
    comm: str
    ppid: int
    is_kthread: bool
    io: Tuple[int, ...]              # PROC_IO_FIELDS order; all zeros if io unreadable
    blkio_ticks: Optional[int]       # delayacct_blkio_ticks (victim signal), None if absent
    io_readable: bool


def parse_proc_stat(text: str) -> Tuple[str, int, int, int, Optional[int]]:
    """Parse /proc/[pid]/stat -> (comm, ppid, flags, start_ticks, blkio_ticks).

    comm can contain spaces and ')' (e.g. "(sd-pam)", "(my )proc)"), so we
    split on the LAST ')'. Fields after it start at field 3 (state); field n
    is at index n-3 in the remainder.
    """
    lp, rp = text.index("("), text.rindex(")")
    comm = text[lp + 1:rp]
    rest = text[rp + 2:].split()
    ppid = int(rest[1])          # field 4
    flags = int(rest[6])         # field 9
    start_ticks = int(rest[19])  # field 22
    blkio = int(rest[39]) if len(rest) > 39 else None  # field 42
    return comm, ppid, flags, start_ticks, blkio


def parse_proc_io(text: str) -> Tuple[int, ...]:
    vals = dict.fromkeys(PROC_IO_FIELDS, 0)
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in vals:
            vals[k] = int(v)
    return tuple(vals[f] for f in PROC_IO_FIELDS)


@dataclass
class ScanStats:
    seen: int = 0
    vanished: int = 0        # exited during scan (expected churn)
    io_denied: int = 0       # /proc/pid/io needs ptrace-read access (root or CAP_SYS_PTRACE)
    malformed: int = 0


def read_process(pid: int, stats: ScanStats) -> Optional[ProcSample]:
    """Read one process; returns None if it disappeared or is unparsable.

    Never raises for expected churn. /proc/[pid]/io is read AFTER stat and the
    starttime is re-checked, so a PID reused between the two reads is caught.
    """
    base = f"/proc/{pid}"
    try:
        comm, ppid, flags, start, blkio = parse_proc_stat(read_file(base + "/stat"))
    except _VANISHED:
        stats.vanished += 1
        return None
    except OSError as e:
        if e.errno in _TRANSIENT_ERRNOS:
            stats.vanished += 1
        else:
            stats.malformed += 1
        return None
    except (ValueError, IndexError):
        stats.malformed += 1  # truncated read while the task was exiting
        return None

    io: Tuple[int, ...] = (0,) * len(PROC_IO_FIELDS)
    readable = False
    try:
        io = parse_proc_io(read_file(base + "/io"))
        readable = True
    except PermissionError:
        stats.io_denied += 1
    except _VANISHED:
        stats.vanished += 1
        return None
    except OSError as e:
        if e.errno in _TRANSIENT_ERRNOS:
            stats.vanished += 1
            return None
        stats.malformed += 1
    except ValueError:
        stats.malformed += 1

    # Guard against PID reuse between the stat and io reads.
    if readable:
        try:
            _, _, _, start2, _ = parse_proc_stat(read_file(base + "/stat"))
            if start2 != start:
                stats.vanished += 1
                return None
        except (OSError, ValueError, IndexError):
            stats.vanished += 1
            return None

    stats.seen += 1
    return ProcSample(ProcKey(pid, start), comm, ppid, bool(flags & PF_KTHREAD),
                      io, blkio, readable)


def scan_processes() -> Tuple[Dict[ProcKey, ProcSample], ScanStats]:
    stats = ScanStats()
    out: Dict[ProcKey, ProcSample] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out, stats
    for name in entries:
        if not name.isdigit():
            continue
        s = read_process(int(name), stats)
        if s is not None:
            out[s.key] = s
    return out, stats


def read_cmdline(pid: int, maxlen: int = 160) -> str:
    """Command line for reporting only (read lazily for top contributors)."""
    try:
        raw = read_file(f"/proc/{pid}/cmdline")
    except OSError:
        return ""
    return raw.replace("\x00", " ").strip()[:maxlen]


def uptime_s() -> float:
    return float(read_file("/proc/uptime").split()[0])


# --------------------------------------------------------------------------
# Process delta tracker (PID reuse, births, exits, reaped-child folding)
# --------------------------------------------------------------------------
@dataclass
class _Debt:
    amount: List[int]
    expires_mono: float


@dataclass
class ProcessTracker:
    """Turns successive process scans into per-interval I/O deltas.

    Correctness rules:
    * Key = (pid, starttime). A reused PID is a new key, never a delta.
    * A key first seen whose start time is AFTER the previous scan was born
      during this interval: its counters started at zero at fork, so the whole
      current value is its delta. Without this, short-lived bursty processes
      would be systematically under-counted.
    * A key that disappears is gone; whatever it did after our last read is
      lost to us — EXCEPT that when the parent reaps it, the kernel folds the
      child's lifetime counters into the parent's cumulative /proc/[ppid]/io
      (wait_task_zombie). The parent then shows a jump containing I/O we
      already credited to the child. We record that as a "debt" against the
      parent and subtract it, then flag the parent row so attribution can say
      "includes reaped children". Debts expire (parent may never reap;
      orphans get reparented) so errors stay bounded.
    """
    debt_ttl_s: float = 10.0
    prev: Dict[ProcKey, ProcSample] = field(default_factory=dict)
    prev_uptime: Optional[float] = None
    debts: Dict[int, _Debt] = field(default_factory=dict)  # by parent pid

    def update(self, cur: Dict[ProcKey, ProcSample], now_mono: float,
               now_uptime: float) -> List[Dict]:
        rows: List[Dict] = []
        n = len(PROC_IO_FIELDS)

        # 1. Exits: register debts against parents for already-credited I/O.
        for key, old in self.prev.items():
            if key not in cur and old.io_readable and any(old.io):
                d = self.debts.get(old.ppid)
                if d is None:
                    self.debts[old.ppid] = _Debt(list(old.io), now_mono + self.debt_ttl_s)
                else:
                    d.amount = [a + b for a, b in zip(d.amount, old.io)]
                    d.expires_mono = now_mono + self.debt_ttl_s
        self.debts = {p: d for p, d in self.debts.items() if d.expires_mono > now_mono}

        # 2. Deltas for live processes.
        for key, s in cur.items():
            if not s.io_readable:
                continue
            old = self.prev.get(key)
            born = False
            if old is not None and old.io_readable:
                delta = [max(0, c - p) for c, p in zip(s.io, old.io)]
                blk = (None if s.blkio_ticks is None or old.blkio_ticks is None
                       else max(0, s.blkio_ticks - old.blkio_ticks))
            elif (self.prev_uptime is not None
                  and key.start_ticks / CLK_TCK >= self.prev_uptime - 1.0 / CLK_TCK):
                delta, blk, born = list(s.io), s.blkio_ticks, True
            else:
                continue  # first sighting of a pre-existing process: baseline only

            reaped_adj = False
            debt = self.debts.get(key.pid)
            if debt is not None and any(delta):
                # Subtract what we already credited to its exited children.
                take = [min(a, b) for a, b in zip(delta, debt.amount)]
                if any(take):
                    delta = [a - b for a, b in zip(delta, take)]
                    debt.amount = [a - b for a, b in zip(debt.amount, take)]
                    reaped_adj = True
                    if not any(debt.amount):
                        del self.debts[key.pid]

            if not any(delta) and not blk:
                continue  # sparse output: only processes that did something
            row = {"pid": key.pid, "start_ticks": key.start_ticks, "comm": s.comm,
                   "ppid": s.ppid, "kthread": int(s.is_kthread),
                   "born_in_interval": int(born), "reaped_child_adjusted": int(reaped_adj),
                   "d_blkio_ticks": blk if blk is not None else ""}
            for i in range(n):
                row["d_" + PROC_IO_FIELDS[i]] = delta[i]
            rows.append(row)

        self.prev = cur
        self.prev_uptime = now_uptime
        return rows
