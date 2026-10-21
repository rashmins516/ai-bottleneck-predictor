# iobn — predictive I/O bottleneck framework (v0.2)

A Linux kernel-telemetry collector, a leakage-safe ML forecaster for device pressure
1.5–3.0 s ahead, "likely contributor" process attribution, and WSL2 validation tools.
**Read `METHODOLOGY.md` first.** It explains every design decision, the evaluation
protocol, and the numbers from the validation run.

```
iobn/
  telemetry.py    Phase 1  /proc/diskstats, /proc/[pid]/io, PSI, meminfo parsers; ProcessTracker
  collector.py    Phase 1  tiered, drift-free sampler -> device.csv / process.csv / meta.json
  features.py     Phase 2  causal window features (numpy prefix sums, shared by train + live)
  labeling.py     Phase 3  device-centric state, future-window labels, purged splits
  model.py        Phase 3  GBDT training, baselines, episode/lead-time eval, live scorer
  attribution.py  Phase 4  likely-contributor ranking + ground-truth validation harness
  wsl.py          Phase 5  WSL2 detection, sector accounting, iostat cross-check, anomaly scan
  workloads.py    burst / db / rogue / decoy workloads with ground truth
  cli.py          python -m iobn {env,wsl-check,collect,experiment,train,attribute-eval,live}
tests/            22 tests, including leakage, feature, threshold and alarm-debounce guards
sample_runs/      4 recorded 300 s runs (device.csv, process.csv, ground_truth.jsonl, meta.json)
results/          train report, attribution report, validation-check output from the sandbox
```

## Quick start

```bash
pip install -r requirements.txt            # numpy, pandas, scikit-learn, joblib, pytest
sudo apt install sysstat                   # for the iostat cross-check
python -m pytest -q tests

# retrain on the included recordings (no root needed)
python -m iobn train sample_runs/all_* --device vda --out model.joblib --report report.json
python -m iobn attribute-eval sample_runs/all_* --model model.joblib

# on your own machine (root, or CAP_SYS_PTRACE, for full /proc/[pid]/io coverage)
sudo $(which python) -m iobn wsl-check --workdir ~/iobn-work
sudo $(which python) -m iobn experiment --scenario all --duration 300 --seed 21 --workdir ~/iobn-work
sudo $(which python) -m iobn live --model model.joblib --duration 120
```

A model is tied to the storage profile it was trained on, so retrain on your WSL2
recordings (the device is usually `sdc`/`sdd`; `iobn env` prints it).

Requires Linux ≥ 4.18 for the full diskstats field set (older kernels work with fewer
fields), and Python ≥ 3.9.
