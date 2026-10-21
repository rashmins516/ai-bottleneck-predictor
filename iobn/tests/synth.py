"""Synthetic diskstats-shaped runs with known episodes (for tests only)."""
import numpy as np
import pandas as pd


def synth_run(seconds=600, dt=0.25, seed=0, n_episodes=14, predictable_frac=0.7):
    rng = np.random.default_rng(seed)
    n = int(seconds / dt)
    ts = 100.0 + np.arange(n) * dt
    iops = rng.normal(60, 5, n).clip(10)
    lat = rng.normal(0.3, 0.05, n).clip(0.05)          # ms
    dirty = rng.normal(0.1, 0.02, n).clip(0)           # MB
    starts = np.sort(rng.choice(np.arange(40, seconds - 30, 1.0), n_episodes, replace=False))
    # enforce spacing so every episode has a quiet lead-in
    keep = [starts[0]]
    for s in starts[1:]:
        if s - keep[-1] > 25:
            keep.append(s)
    episodes = []
    for s in keep:
        dur = rng.uniform(4, 9)
        predictable = rng.random() < predictable_frac
        i0, i1 = int(s / dt), int((s + dur) / dt)
        if predictable:  # Dirty page build-up + mild latency creep 4 s before onset
            p0 = int((s - 4) / dt)
            ramp = np.linspace(0, 1, i0 - p0)
            dirty[p0:i0] += 600 * ramp
            iops[p0:i0] += 400 * ramp
            lat[p0:i0] += 0.4 * ramp
        iops[i0:i1] = rng.normal(5000, 300, i1 - i0)
        lat[i0:i1] = rng.normal(25, 4, i1 - i0).clip(5)
        dirty[i0:i1] = np.linspace(600 if predictable else 50, 0, i1 - i0)
        episodes.append((ts[i0], predictable))
    ios = np.round(iops * dt)
    rd = np.round(ios * 0.5)
    wr = ios - rd
    ticks = np.round(ios * lat)
    psi = np.where(lat > 5, 0.8, 0.01) * dt * 1e6
    df = pd.DataFrame({
        "ts": ts, "wall": 1.7e9 + ts, "dt_s": dt, "device": "sdx", "valid": 1, "gap": 0, "suspend": 0,
        "d_rd_ios": rd, "d_rd_merges": 0, "d_rd_sectors": rd * 8, "d_rd_ticks": np.round(ticks * 0.5),
        "d_wr_ios": wr, "d_wr_merges": 0, "d_wr_sectors": wr * 8, "d_wr_ticks": np.round(ticks * 0.5),
        "d_io_ticks": np.minimum(ticks, dt * 1000), "d_time_in_queue": ticks,
        "d_dc_ios": 0, "d_dc_merges": 0, "d_dc_sectors": 0, "d_dc_ticks": 0, "d_fl_ios": 0, "d_fl_ticks": 0,
        "in_flight": np.round(iops * lat / 1000), "d_psi_some_us": psi, "d_psi_full_us": psi * 0.8,
        "iowait_frac": 0.0, "steal_frac": 0.0, "mem_dirty_kb": dirty * 1024, "mem_writeback_kb": 0,
    })
    for c in ("r_iops", "w_iops", "r_bps", "w_bps", "r_await_ms", "w_await_ms", "util", "aqu_sz"):
        df[c] = np.nan
    return df, episodes
