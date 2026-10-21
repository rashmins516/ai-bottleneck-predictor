"""Phase 1 tests: parsing, counter arithmetic, process churn and identity."""
import os
import subprocess
import sys
import threading

import pytest

from iobn import telemetry as T

LINE_11 = "   8       0 sda 100 2 800 50 200 3 1600 90 1 120 140"
LINE_15 = LINE_11 + " 5 0 40 2"
LINE_17 = LINE_15 + " 7 9"


@pytest.mark.parametrize("line,n_present", [(LINE_11, 11), (LINE_15, 15), (LINE_17, 17)])
def test_parse_diskstats_variable_width(line, n_present):
    d = T.parse_diskstats(line + "\n garbage line\n")
    v = d["sda"]
    assert len(v) == T.N_DISKSTAT
    assert v[T.IDX["rd_ios"]] == 100 and v[T.IDX["time_in_queue"]] == 140
    assert all(x == 0 for x in v[n_present:])


def test_parse_diskstats_filters_and_skips_malformed():
    text = LINE_17 + "\n   8 1 sda1 x y z 1 2 3 4 5 6 7 8\n"
    assert set(T.parse_diskstats(text)) == {"sda"}
    assert T.parse_diskstats(text, devices=["nvme0n1"]) == {}


def test_counter_delta_wrap_and_reset():
    assert T.counter_delta(150, 100, "rd_ios") == 50
    # 32-bit ms field wrap
    assert T.counter_delta(10, T.U32 - 5, "rd_ticks") == 15
    # non-ms counter going backwards = reset, never silently 0
    assert T.counter_delta(10, 500, "rd_ios") is None


def test_device_interval_latency_is_device_level():
    prev = T.parse_diskstats(LINE_17)["sda"]
    cur = list(prev)
    cur[T.IDX["rd_ios"]] += 100
    cur[T.IDX["rd_ticks"]] += 250       # 2.5 ms per read
    cur[T.IDX["rd_sectors"]] += 800
    cur[T.IDX["io_ticks"]] += 125
    cur[T.IDX["time_in_queue"]] += 250
    m = T.device_interval(prev, tuple(cur), 0.25)
    assert m["r_await_ms"] == pytest.approx(2.5)
    assert m["r_bps"] == pytest.approx(800 * 512 / 0.25)
    assert m["util"] == pytest.approx(0.5)
    assert m["aqu_sz"] == pytest.approx(1.0)
    assert m["w_await_ms"] != m["w_await_ms"]  # NaN when no writes completed
    cur[T.IDX["wr_ios"]] = prev[T.IDX["wr_ios"]] - 1
    assert T.device_interval(prev, tuple(cur), 0.25) is None  # reset -> invalid sample


def test_parse_proc_stat_hostile_comm():
    rest = ["0"] * 40          # rest[i] is /proc/pid/stat field i+3
    rest[0], rest[1], rest[6], rest[19], rest[39] = "S", "1", str(T.PF_KTHREAD), "98765", "42"
    text = "1234 (my )proc) x) " + " ".join(rest)
    comm, ppid, flags, start, blkio = T.parse_proc_stat(text)
    assert comm == "my )proc) x"
    assert ppid == 1 and start == 98765 and blkio == 42
    assert flags & T.PF_KTHREAD


def test_read_process_vanished_returns_none():
    p = subprocess.Popen(["true"])
    p.wait()
    st = T.ScanStats()
    assert T.read_process(p.pid, st) is None
    assert st.vanished == 1


def test_scan_survives_heavy_churn():
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            subprocess.run([sys.executable, "-c", "pass"])

    th = [threading.Thread(target=churn) for _ in range(3)]
    for t in th:
        t.start()
    try:
        for _ in range(30):
            procs, st = T.scan_processes()   # must never raise
            assert st.seen > 0
    finally:
        stop.set()
        for t in th:
            t.join()


def _sample(pid, start, io, ppid=1):
    return T.ProcSample(T.ProcKey(pid, start), "x", ppid, False, tuple(io), None, True)


def test_tracker_pid_reuse_birth_and_reaped_child():
    tr = T.ProcessTracker()
    z = [0] * 7
    parent = _sample(10, 100, z)
    child = _sample(20, 500, [0, 0, 0, 0, 0, 1000, 0], ppid=10)
    # first scan: baseline only
    assert tr.update({parent.key: parent, child.key: child}, 1.0, 10.0) == []
    # child writes 4000 more bytes
    child2 = _sample(20, 500, [0, 0, 0, 0, 0, 5000, 0], ppid=10)
    rows = tr.update({parent.key: parent, child2.key: child2}, 2.0, 11.0)
    assert rows[0]["d_write_bytes"] == 4000
    # child exits & is reaped: parent's cumulative jumps by child's lifetime 5000 (+200 own)
    parent3 = _sample(10, 100, [0, 0, 0, 0, 0, 5200, 0])
    # PID 20 reused by a NEW process born after the previous scan (uptime 11.0 s)
    reborn = _sample(20, int(11.5 * T.CLK_TCK), [0, 0, 0, 0, 0, 300, 0], ppid=10)
    rows = {r["pid"]: r for r in tr.update({parent3.key: parent3, reborn.key: reborn}, 3.0, 12.0)}
    assert rows[10]["d_write_bytes"] == 200          # child's already-credited bytes removed
    assert rows[10]["reaped_child_adjusted"] == 1
    assert rows[20]["d_write_bytes"] == 300          # new process: full counters, not 300-5000
    assert rows[20]["born_in_interval"] == 1
