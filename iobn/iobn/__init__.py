"""iobn — predictive I/O bottleneck framework (kernel telemetry + ML).

Phase map
---------
Phase 1  telemetry.py, collector.py   low-overhead procfs ingestion
Phase 2  features.py                  causal sliding-window device-pressure features
Phase 3  labeling.py, model.py        future-window labels, purged splits, evaluation
Phase 4  attribution.py               "likely contributor" ranking + ground-truth validation
Phase 5  wsl.py                       WSL2 telemetry validation; workloads.py drives all phases
"""

__version__ = "0.2.0"
