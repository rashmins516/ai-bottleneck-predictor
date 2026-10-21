# Predictive I/O Bottleneck Framework — Revised Methodology (v0.2)

This document replaces Sections III–VI of the draft paper's methodology. Each phase lists
the review finding it fixes, the design, and the module that implements it.

```
 workloads.py ──► (kernel) ──► Phase 1 telemetry.py / collector.py
                                   │  device.csv  (device-level latency & pressure)
                                   │  process.csv (process-level throughput only)
                                   ▼
                          Phase 2 features.py      causal sliding windows, no queue-overflow assumption
                                   ▼
                          Phase 3 labeling.py      y(t) from (t+1.5 s, t+3.0 s] only; purged splits
                                  model.py         GBDT + persistence & %util baselines; lead time
                                   ▼
                          Phase 4 attribution.py   "likely contributors" + ground-truth hit rate
                          Phase 5 wsl.py           WSL2 validation: iostat cross-check, sector accounting,
                                                   timing-anomaly scan
```

---

## Phase 1 — Telemetry ingestion (`iobn/telemetry.py`, `iobn/collector.py`)

**Finding fixed.** The draft treated `/proc/diskstats` and `/proc/[pid]/io` as if they
provided per-process queue wait times. They do not.

**What each source actually provides**

| Source | Granularity | Meaning | Used as |
|---|---|---|---|
| `/proc/diskstats` `rd_ticks`, `wr_ticks` (ms) | per device | total time completed requests spent between block-layer start and completion | device latency = Δticks / Δios |
| `/proc/diskstats` `time_in_queue` (ms) | per device | integral of in-flight requests over time | avg in-flight `aqu = Δtime_in_queue / Δt` |
| `/proc/diskstats` `io_ticks` | per device | time with ≥1 request in flight | `util` (advisory only on SSD/NVMe) |
| `/proc/pressure/io` `total=` (µs) | system | time some/all tasks stalled on I/O | stall share per interval |
| `/proc/meminfo` `Dirty`, `Writeback` | system | page-cache data not yet on disk | leading indicator for writeback |
| `/proc/[pid]/io` | per process | bytes/syscalls charged to the task | process **throughput** only |
| `/proc/[pid]/stat` field 42 | per process | `delayacct_blkio_ticks`, needs `kernel.task_delayacct=1` | **victim** signal, never blame |

There is no per-process queue wait anywhere in procfs. The two metric families are written
to different files (`device.csv`, `process.csv`) and are never mixed in one column.

**Parser correctness**
- Sectors in diskstats are always 512 bytes, independent of the device's block size.
- 11/15/17-field kernels are all accepted; malformed lines are skipped, not fatal.
- The millisecond fields are printed as 32-bit (`%u`) even on 64-bit kernels, so they wrap
  at 2³² ms. `counter_delta()` handles that wrap. A counter that goes backwards for any other
  reason (device re-added) invalidates the sample instead of producing a fake zero.
- Raw deltas are stored, so latency can be aggregated correctly later as Σticks/Σios.

**Short-lived processes** (`ProcessTracker`)
- Every per-PID read tolerates `FileNotFoundError`, `ProcessLookupError`/`ESRCH` (exit
  between open and read), `PermissionError` (needs root or `CAP_SYS_PTRACE`), and
  truncated content. These are counted as coverage statistics, not raised.
- Processes are keyed by `(pid, starttime)`. A reused PID becomes a new key, so it never
  produces a bogus delta. Start time is re-checked after reading `io` to catch reuse
  between the two reads.
- A process born during the interval has counters that started at zero at fork, so its
  full value is its delta. Without this rule, bursty short-lived workers are undercounted.
- When a parent reaps a child, the kernel adds the child's lifetime I/O to the parent's
  cumulative counters. The tracker subtracts what it already credited to the child, so it
  is not counted twice, and flags the parent row `reaped_child_adjusted`.

**Overhead** is measured, not assumed. `meta.json` records collector CPU time from
`getrusage` as a % of one core and of total capacity, plus mean and max tick work time.
Device files are read every tick (default 250 ms). The O(#processes) `/proc` scan runs on a
slower tier (`proc_every`). The draft's "O(1)" claim should be restated: device ingestion
is O(#devices) per tick and process ingestion is O(#processes) per process tick.

**Clock hygiene.** Timestamps use `CLOCK_MONOTONIC`. Sleeps target absolute deadlines, so
there is no drift. A tick is flagged `gap` if dt > 2× the interval, and `suspend` if
`CLOCK_BOOTTIME − CLOCK_MONOTONIC` jumps. Flagged ticks are excluded from every window
and label.

---

## Phase 2 — Feature engineering without queue-overflow assumptions (`iobn/features.py`)

**Finding fixed.** blk-mq NVMe/SSD stacks spread requests over many deep hardware queues,
so they essentially never "overflow". Near saturation, three things happen instead:
latency rises, throughput stops scaling with offered concurrency, and tasks stall.

**Features.** All windows are right-aligned time windows (t−w, t], for w ∈ {1, 5, 30} s.

| Group | Features | Why |
|---|---|---|
| Latency | `await_w = Σ(rd_ticks+wr_ticks)/Σ(rd_ios+wr_ios)`, read/write split, `lat_accel = await_1s/await_30s`, `lat_trend` (slope of await_1s over 5 s) | I/O-weighted, so it is not distorted by near-idle ticks. ms resolution makes per-tick values unusable on fast or virtual disks. |
| Throughput | `iops_w`, `bps_w`, `tput_trend_mbps`, `avg_req_kb_w`, `write_byte_frac_w` | load level and mix |
| Scaling | `aqu_w`, `aqu_trend`, **`scaling_elasticity` = d log IOPS / d log aqu** over 5 s | By Little's law, aqu = IOPS × await. An elasticity near 1 means added concurrency still buys throughput; near 0 it only buys latency. This is the saturation signature that replaces "queue overflow". |
| Stall | `psi_some_w`, `psi_full_w` (from cumulative totals, not the kernel's pre-smoothed avg10), `iowait_w` | measured task stalls |
| Writeback | `dirty_mb`, `writeback_mb`, `dirty_trend_mbps` (5 s and 1 s) | Buffered writes accumulate before the device sees them. This is one of the few genuinely *leading* signals. |
| Context | `util_w`, `flush_rate_w`, `merge_ratio_w`, `steal_w`, `in_flight` | `util` is kept as a feature but never used as the label (100% busy ≠ saturated on parallel devices) |
| Instability / dynamics | `await_volatility_ms_w`, `throughput_volatility_mbps_w`, latency/queue/throughput second derivatives | Rolling standard deviation and causal derivatives distinguish a stable high load from rapidly worsening, bursty pressure. |
| Process coupling | physical and logical process read/write rates, top-process share, process/device throughput ratios, rolling pressure/rate correlation | Physical rates use `read_bytes`/`write_bytes` (minus cancelled writes); `rchar`/`wchar` syscall rates remain separate. Sparse scans are forward-aligned only from the latest scan (maximum age 2 s); stale values become NaN. |

**Causality.** Every window is (t − w, t]. Window sums come from prefix sums plus
`searchsorted`, so there is no centring, no negative shift and no backfill. The test
`tests/test_leakage.py::test_features_are_strictly_causal` overwrites every row after t
with random values and asserts that features up to t are bit-identical.

**One engine for training and serving.** `device_features_arrays()` is a numpy
prefix-sum engine. Training and the live predictor call the same function, so there
is no train/serve skew. It matched pandas' right-closed time-window `rolling()`
exactly on a recorded run (max |Δ| = 0 for `await_5s`, 9e-10 relative for trend slopes).
It computes a whole 1,232-row run in ~10 ms and scores one live tick in ~3 ms,
including the model.

`BaselineNormalizer` adds scale-free ratios (latency relative to baseline latency) so a
model can transfer across storage profiles. It is fit on **training rows only**.

---

## Phase 3 — True predictive labelling and leakage prevention (`iobn/labeling.py`, `iobn/model.py`)

**Finding fixed.** If the label overlaps the feature window, the model learns to *detect*
an existing bottleneck and scores well while forecasting nothing.

**Pressure state** (computed only from data in (u−1 s, u], debounced over 0.5 s). The
state is device-centric:

```
pressure(u) = await_1s(u) ≥ max(floor_ms, k · baseline_ms)  AND  iops_1s(u) ≥ min_iops
```

- `baseline_ms` is the median **1 s** I/O-weighted latency over busy *training* rows.
  An earlier version used the 30 s window. With frequent episodes, an I/O-weighted
  30 s average is dominated by the episode inside it, so its median drifted to
  "episode latency" (20.8 ms on synthetic data) and the threshold stopped firing.
- Defaults are k = 3, floor = 2 ms and min_iops = 50. Tune the floor per profile.

**Why PSI is not a label criterion (a leakage path we found in our own data).** The
first version used `… OR psi_some_1s ≥ 0.25`. PSI io.some means "at least one task
waited on I/O". It saturates whenever *any* process runs a back-to-back synchronous
I/O loop, even on a healthy device.

In our recordings, random-I/O workloads at 10–15 k IOPS kept device latency at
0.2–0.3 ms, but still pushed PSI to ~0.9. Every workload start therefore became a
"bottleneck". The model then "predicted" it from rising IOPS ~1 s before the debounced
state caught up. That run reported 100% of episodes warned ahead with a 2.75 s median
lead, and it was detection, not forecasting.

PSI is now a feature only. `psi_mode="and"` can optionally require it as
corroboration ("latency up AND tasks stalling"). `"or"` is kept only to reproduce the
failure.

**Future label.**

```
y(t) = 1  ⇔  ∃u ∈ (t + 1.5 s, t + 3.0 s] : pressure(u) = 1
```

- **No shared data.** The raw data behind y(t) lies in (t + 1.5 − 1.0 − 0.5, t + 3.0] =
  (t, t + 3.0], which is strictly after the feature data (≤ t). `make_labels` raises if
  `h_start < state_window + debounce`.
- **Incomplete futures are dropped.** If the label window runs past the end of the run,
  or crosses a gap or suspend, y = NaN and the row is dropped. It is never set to 0.
- **Onset rows.** These are rows with no pressure in [t − 3 s, t]. The check uses
  both the debounced state and the single-tick raw state (`await_tick` over the
  threshold).
  - The debounced 1 s state lags the first degraded tick by up to ~1 s, so without the
    raw check some "onset" rows would already sit on a degraded device.
  - On onset rows persistence cannot score, so this is where forecasting is proven.
- **Lead time is measured from the raw onset**, i.e. the first degraded tick. It is
  *not* measured from the debounced episode start, because that would count the
  label's own lag as forecast. The report prints both, and marks the second as
  "do not report".

**Splits** (`purged_split`)
- With several recordings, whole runs are held out (latest first).
- Within one run, the split is a chronological cut at c with a two-sided purge. Train rows
  need t + 3.0 < c, and test rows need t − 30 s ≥ c (the maximum feature lookback). No raw
  sample contributes to both a training label and a test feature.
- The alarm threshold is chosen from **out-of-fold** predictions *inside* the training
  set, never on test.
  - With ≥ 2 training runs, it uses leave-one-run-out; with one run, a purged
    validation cut.
  - The default policy maximizes debounced recall subject to at least 0.80 precision;
    `--threshold-objective f1` selects the alternative maximum-F1 policy.
    If the floor is infeasible in OOF predictions, the report says so and uses the
    best-F1 fallback. Threshold selection is reset across runs and gaps.
  - Debouncing uses `--alarm-consecutive` consecutive evaluation scores (default 2),
    and the same policy is used for offline metrics and live alerts.
- `purged_walk_forward` gives expanding-window CV with the same purge, for hyperparameter
  search.

**Model.** `HistGradientBoostingClassifier` with balanced class weights and native NaN
handling (latency is legitimately undefined at idle). `RandomForest` is available as an
alternative (`--model rf`).

**Evaluation report** (`train` writes it as JSON)
1. **Row level.** Precision, recall, F1, ROC-AUC and PR-AUC on **all** rows and on
   **onset** rows, for three predictors:
   - `model`
   - `persistence`: "pressure now" using the fastest evidence, raw tick OR debounced state
   - `static_util90`: a reactive %util alarm
2. **Episode level.** The alarm policy is the same as live: 2 consecutive ticks over
   the threshold. Reported:
   - share of episodes (preceded by ≥ 3 s calm) warned in [raw onset − 3 s, raw onset)
   - lead-time p25, median and p75 against the raw onset
   - near-miss alarms (raw degradation followed, but below the episode criteria)
   - false alarms per hour (nothing followed)
3. **Timing against ground truth.** First alarm and raw onset are reported relative to
   the injected workload's start. This separates "forecast from a precursor" from
   "noticed the workload starting".
4. **Permutation importance** (drop in PR-AUC). If the top features are all
   `await_1s`/`psi_1s`, the model is a detector, not a forecaster.

The draft's "1.5–3.0 s lead time" becomes the **design horizon**. The paper should report
the *measured* lead-time distribution and warned-ahead fraction for each workload class.
Abrupt onsets (a fully parallel job starting at once) are inherently unpredictable 1.5 s
ahead and should be reported separately rather than hidden in an average.

---

## Phase 4 — Process attribution as "likely contributors" (`iobn/attribution.py`)

**Finding fixed.** Process byte counters and device bytes do not map 1:1:
- `write_bytes` is charged when a page is **dirtied**, not when writeback hits the device.
- Flusher and journal threads (`kworker`, `jbd2`) do the actual device I/O on behalf of
  others.
- `cancelled_write_bytes` never reaches disk.
- `rchar` counts page-cache hits, pipes and sockets.
- Small synchronous I/O causes queueing far beyond its byte share.

**Signals per process** over the last 10 s (configurable):

| Signal | Definition | Weight |
|---|---|---|
| physical byte share | (read_bytes + write_bytes − cancelled) / Σ over user processes | 0.5 |
| estimated physical-op share | min(syscalls × physical/logical, physical bytes / 4 KiB) — the cap stops chatty loggers that barely touch the disk from looking like small-I/O storms | 0.2 |
| latency coupling | max Spearman ρ between the process's physical rate and device await, with the process leading by 0–2 ticks | 0.3 |

- Kernel threads are listed as **kernel I/O proxies** and never ranked.
- Processes with high `delayacct_blkio_ticks` are listed as **likely victims**.
- The report includes the **unattributed residual** (device bytes not explained by process
  counters).
- A process must own ≥ 2% of bytes or ops to be listed at all. Coupling alone never
  qualifies it.
- The collector excludes itself.
- Confidence is *high* only when the top candidate is separated from the next one, coupled
  (ρ ≥ 0.4), owns ≥ 40% of attributed bytes, and the residual is ≤ 50%.
- Confidence is never better than *low* when the ranking explains < 20% of the device's
  bytes.
- Notes flag pending writeback, large residuals and reaped-child absorption.

Every log line says **"LIKELY CONTRIBUTORS"** and ends with: *attribution is correlational;
candidates for throttling review, not proven causes.*

**Cross-validation with known workloads** (`validate_against_ground_truth`,
`iobn attribute-eval`). The workload generator logs every injected process with
`(pid, starttime, t_start, t_end, role, expected_contributor)`. At every pressure-episode
start, plus an equal number of random calm times as controls, the harness reports:

- **hit@1, hit@3 and MRR** for the deliberately injected culprit
- **confident-accusation rate** in control windows (false blame)
- **hit@1 broken down by confidence label** (does "high" mean anything?)

The **decoy** role (huge `rchar`, ~0 `read_bytes`) checks that attribution is cache-aware.
These hit rates, not the ranking itself, are what the paper can claim.

---

## Phase 5 — WSL2 telemetry validation (`iobn/wsl.py`, `iobn wsl-check`)

**What changes under WSL2**
- The distro's root filesystem is an ext4 image on a Hyper-V virtual SCSI disk (usually
  `sdc` or `sdd`). diskstats sees the **guest** block layer only. Host caching makes
  latencies look optimistic, with spikes when the host flushes.
- `/mnt/c` and other Windows drives use 9p/drvfs/virtiofs. Their I/O **never appears in
  `/proc/diskstats`**. `check_workdir` fails if workloads would run there.
- WSL1 has no guest block layer, so its telemetry is meaningless (detected; the check
  stops).
- Steal time, VM pauses while Windows sleeps (BOOTTIME jump) and host time sync (wall-clock
  steps) all distort time-based metrics.
- PSI and delay accounting depend on the WSL kernel build, so the code detects them
  rather than assuming them.

**Validation steps**
1. `environment()`: kernel, WSL version, root device, Windows mounts, PSI, delayacct,
   ptrace access, CLK_TCK.
2. `sector_accounting_check`: write and read back N MB with `O_DIRECT`. The Δsectors in
   diskstats must match. Writes pass at 0.95–1.6× (journal overhead); reads at 0.95–1.3×.
   This proves the workload directory is visible to the collector.
3. `iostat_crosscheck`: runs `iostat -dxy -o JSON <dev> 1 N` alongside our counters,
   under a controlled load, and compares run-level totals (r/w IOPS, kB, I/O-weighted
   await, %util). Status is PASS at ≤ 10% relative error. iostat reads the same kernel
   file, so disagreement means a parsing or delta bug, or window skew.
4. `timing_anomaly_scan` (runs automatically after every `experiment`) reports:
   - gap and suspend rows
   - %util > 100% rows
   - share of busy ticks with 0 ms total latency (ms quantisation / host cache)
   - p99/p50 await ratio (host flush bimodality)
   - steal mean and max
   - wall-clock steps

**Adjustments made for WSL**
- Latency is window-aggregated, never per tick.
- Invalid ticks are excluded from windows *and* from label windows.
- Baselines and thresholds are fit per storage profile from training data.
- `util` is advisory only.
- Workloads use O_DIRECT where supported, and fall back to buffered I/O +
  `fadvise(DONTNEED)` with the fallback recorded.
- Files are pre-filled with real data, never `fallocate`d, because O_DIRECT reads of
  unwritten extents never touch the device.

**Reporting rule.** WSL2 numbers describe the virtual disk. Present them as a
development-environment result, and reproduce the headline numbers on native Linux, or
state that they were not reproduced.

---

## Pipeline validation run (development sandbox, not WSL)

**Platform.** The pipeline was exercised end to end on a Linux 6.18 VM with 2 vCPUs, a
virtio disk (`vda`, ext4), PSI available and delayacct off. This is **not** the team's
WSL2 environment. These numbers validate the pipeline and show what to report; they are
not results for the paper. Re-run every step on WSL2 and, for headline numbers, on
native Linux.

**Data.** Four 300 s recordings of the `all` scenario (seeds 21–24), in `sample_runs/`.
Training used seeds 21–23 and testing the held-out run, seed 24.

### Phase 1 and Phase 5 checks

| Check | Result |
|---|---|
| iostat cross-check (8 s, controlled load) | worst relative error 0.2% across r/w IOPS, kB, await, %util → PASS |
| Sector accounting, 128 MiB O_DIRECT | write ratio 1.00, read ratio 1.00 → PASS |
| Collector CPU (250 ms device tier, 500 ms process tier, ~70 processes) | 1.07–1.19% of one core (0.5–0.6% of total); 0 overruns in 4,928 ticks |
| Live: collector + scoring every 0.5 s + attribution on alarms | 3.2% of one core (1.6% of total) |
| Timing-anomaly scan | WARN on every run: await p99/p50 = 287–380 (bimodal: writeback spikes); steal up to 34% in one tick |

### Phase 3 — Forecasting on the held-out run

There were 6 eligible episodes, which is a small sample.

| Predictor | Onset rows: precision / recall | All rows F1 | Episodes warned before raw onset | Lead vs raw onset |
|---|---|---|---|---|
| model (GBDT, p* = 0.38) | 1.00 / 0.93 | 0.77 | 4 / 6 | median 1.9 s (IQR 1.4–2.1 s) |
| persistence | 0 / 0 | 0.69 | n/a | n/a |
| static %util ≥ 90% | 0 / 0 | 0.00 | n/a | n/a |

- False alarms were 1 per ~5 min (≈ 12/h), with 2 near-misses.
- The top features by permutation importance were `dirty_mb`, `scaling_elasticity`,
  `dirty_trend_1s_mbps` and `dirty_trend_mbps`.
- The first alarm came a median 2.2 s after the burst process started, and ~10 s before
  the device degraded.

**How to read this.** On this virtio disk, only **writeback storms** produced device-level
pressure. Mixed random-I/O and fsync-heavy "rogue" workloads at 10–15 k IOPS kept latency
at 0.2–0.3 ms. Those processes were I/O-bound, but the device was not saturated.

The forecast signal is therefore the page-cache build-up that precedes writeback: a
genuine, physically meaningful precursor, but only one bottleneck class. The paper must
not generalise to random-I/O saturation without data from a device that saturates under
it (the WSL2 vhdx, a SATA SSD, or an HDD).

### Phase 4 — Attribution against ground truth

The harness queried 66 times: 40 culprit windows at episode starts and 26 calm controls.

| Metric | Value |
|---|---|
| hit@1 / hit@3 / MRR | 0.975 / 0.975 / 0.975 |
| hit@1 by culprit role | burst 18/18, db 12/12, rogue 9/10 |
| Wrong answer given with high or medium confidence | 0% |
| hit@1 by confidence label | high 1.00 (n = 3), medium 1.00 (n = 23), low 0.93 (n = 14) |
| Decoy (page-cache reader, huge `rchar`) ranked in top 3 | 0 of 6 windows |
| Median unattributed residual | 0.6% |

These culprits were dominant and unambiguous. Expect lower hit rates when several real
applications share the device, which is why the confidence breakdown is reported.

### Claims in the draft paper that this changes

| Draft claim | Status |
|---|---|
| "queues overflow" | Replace with "latency rises and throughput stops scaling" (Phase 2) |
| O(1) scalability | Device ingestion is O(#devices) per tick; the process scan is O(#processes) per process tick |
| "< 1.5% of CPU" | Now measured: ~1.1% of one core for the collector and ~3.2% with live inference on 2 vCPUs. Report per platform. |
| "1.5–3.0 s lead time" | This is the design horizon. Report the measured lead against raw onset, per bottleneck class. |
| PID isolation | Replace with "likely contributors" plus the ground-truth hit rate |

---

## Reproducing

```bash
pip install -r requirements.txt && sudo apt install sysstat
python -m pytest -q tests                                   # 18 tests incl. leakage guards
sudo python -m iobn wsl-check --workdir ~/iobn-work         # Phase 5
for s in 21 22 23 24; do sudo python -m iobn experiment --scenario all --duration 300 \
     --seed $s --workdir ~/iobn-work --out-root runs; done   # recordings + ground truth
DEV=$(python -c "from iobn.telemetry import device_for_path as d; import os; print(d(os.path.expanduser('~/iobn-work')))")
python -m iobn train runs/all_* --device $DEV --out model.joblib --report report.json
python -m iobn attribute-eval runs/all_s23_* runs/all_s24_* --model model.joblib
sudo python -m iobn live --model model.joblib --duration 120   # prints likely-contributor logs
```

To retrain on the included recordings without collecting anything:
`python -m iobn train sample_runs/all_* --device vda --report report.json`.

Run collection as root, or grant `CAP_SYS_PTRACE`. Otherwise `/proc/[pid]/io` is
unreadable for other users' processes, and `meta.json` reports the coverage loss.
To get the victim signal, enable `sudo sysctl kernel.task_delayacct=1`.

**WSL2 runbook**
1. Check the WSL version with `wsl -l -v` in PowerShell. Version 2 is required.
2. Work inside the distro's ext4 filesystem (`~/iobn-work`), never under `/mnt/c`.
   `iobn env` shows which block device that maps to (typically `sdc`/`sdd`); pass it
   as `--device` to `train`.
3. Run `sudo $(which python) -m iobn wsl-check --workdir ~/iobn-work` and keep the JSON
   with your results. Sector accounting and the iostat cross-check must be PASS.
4. Don't let Windows sleep during recordings. Suspends are detected and excluded, but
   they waste data.
5. Expect the WARN for latency bimodality, since host-side flushes are normal under
   Hyper-V. Set `--lat-floor-ms` from the measured baseline (`await_p50_ms` in the
   anomaly scan) rather than the 2 ms default.
