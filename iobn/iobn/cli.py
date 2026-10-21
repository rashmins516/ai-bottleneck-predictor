"""Command-line entry point: python -m iobn <command> ...

  env            environment summary (kernel, WSL, devices, PSI, delayacct)
  wsl-check      Phase 5 validation suite (sector accounting + iostat cross-check)
  collect        Phase 1 collector -> run dir (device.csv, process.csv, meta.json)
  experiment     collector + reproducible workload scenario + ground truth + anomaly scan
  train          Phases 2-3: features, future labels, purged split, model + report
  attribute-eval Phase 4: attribution hit-rate against workload ground truth
  live           collector + model + attribution, printing likely-contributor logs
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import deque


def _dump(obj):
    print(json.dumps(obj, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)))


def cmd_env(a):
    from .wsl import check_workdir, environment
    _dump({"environment": environment(), "workdir": check_workdir(a.workdir)})


def cmd_wsl_check(a):
    from .wsl import run_wsl_validation
    rep = run_wsl_validation(a.workdir, a.iostat_seconds)
    _dump(rep)
    statuses = [v.get("status") for v in rep.values() if isinstance(v, dict) and "status" in v]
    sys.exit(1 if "FAIL" in statuses or "fatal" in rep else 0)


def cmd_collect(a):
    from .collector import Collector, CollectorConfig
    cfg = CollectorConfig(devices=a.devices, interval_s=a.interval, proc_every=a.proc_every)
    meta = Collector(cfg, out_dir=a.out).run(a.duration)
    _dump(meta)


def _scenario_child(scenario, workdir, duration, gt_path, seed, data_mb):
    from .workloads import run_scenario
    run_scenario(scenario, workdir, duration, gt_path, seed=seed, data_mb=data_mb)


def cmd_experiment(a):
    from .collector import Collector, CollectorConfig
    from .telemetry import device_for_path
    from .wsl import check_workdir, timing_anomaly_scan
    chk = check_workdir(a.workdir if os.path.isdir(a.workdir) else os.path.dirname(os.path.abspath(a.workdir)))
    if chk["status"] == "FAIL":
        sys.exit(chk["detail"])
    os.makedirs(a.workdir, exist_ok=True)
    dev = a.device or device_for_path(a.workdir)
    run_dir = os.path.join(a.out_root, f"{a.scenario}_s{a.seed}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    gt_path = os.path.join(run_dir, "ground_truth.jsonl")
    # Prefill outside the measured window so it does not create a fake episode.
    from .workloads import prefill
    prefill(os.path.join(a.workdir, "db.dat"), a.data_mb)
    prefill(os.path.join(a.workdir, "rogue.dat"), max(64, a.data_mb // 4))
    prefill(os.path.join(a.workdir, "decoy.dat"), 64)
    os.sync()          # flush unrelated dirty data so the run starts from a clean device state
    time.sleep(3)
    child = mp.get_context("fork").Process(
        target=_scenario_child, args=(a.scenario, a.workdir, a.duration, gt_path, a.seed, a.data_mb))
    cfg = CollectorConfig(devices=[dev], interval_s=a.interval, proc_every=a.proc_every)
    col = Collector(cfg, out_dir=run_dir)
    child.start()
    meta = col.run(a.duration + 8)
    child.join(timeout=30)
    meta["scenario"] = {"name": a.scenario, "seed": a.seed, "duration_s": a.duration,
                        "workdir": a.workdir, "device": dev}
    meta["timing_anomalies"] = timing_anomaly_scan(os.path.join(run_dir, "device.csv"), dev)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(run_dir)
    _dump({"overhead": meta["overhead"], "coverage": meta["coverage"],
           "timing_anomalies": meta["timing_anomalies"]})


def cmd_train(a):
    from .features import FeatureConfig
    from .labeling import BottleneckCriteria, LabelConfig
    from .model import TrainConfig, train
    cfg = TrainConfig(device=a.device, model=a.model, test_frac=a.test_frac,
                      min_precision=a.min_precision, alarm_consecutive=a.alarm_consecutive,
                      threshold_objective=a.threshold_objective,
                      labels=LabelConfig(h_start_s=a.h_start, h_end_s=a.h_end),
                      criteria=BottleneckCriteria(lat_floor_ms=a.lat_floor_ms,
                                                  lat_multiplier=a.lat_multiplier,
                                                  min_iops=a.min_iops))
    rep = train(a.runs, cfg, out_path=a.out)
    _dump(rep)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=2, default=str)


def cmd_attribute_eval(a):
    import joblib
    import numpy as np
    from .attribution import AttributionConfig, validate_against_ground_truth
    from .features import build_features_for_runs
    from .labeling import episodes, make_labels
    from .model import load_runs
    b = joblib.load(a.model)
    dev_all, proc_all, gt = load_runs(a.runs)
    if gt.empty:
        sys.exit("runs have no ground_truth.jsonl (record them with `experiment`)")
    F = build_features_for_runs(dev_all, b["device"], b["feature_cfg"])
    q = {}
    rng = np.random.default_rng(0)
    for rid, g in F.groupby("run_id", sort=False):
        L = make_labels(g, g["ts"].to_numpy(), b["criteria"], b["label_cfg"])
        ts = g["ts"].to_numpy()
        eps = episodes(ts, L["state_now"].to_numpy())
        # query at each episode start (+0.5 s) and at random quiet times (controls)
        times = [s + 0.5 for s, _ in eps]
        quiet = ts[(L["state_now"].fillna(0).to_numpy() == 0) & (ts > ts[0] + 15)]
        if len(quiet):
            times += list(rng.choice(quiet, size=min(len(eps), len(quiet)), replace=False))
        q[rid] = sorted(times)
    res = validate_against_ground_truth(proc_all, dev_all, gt, b["device"], q,
                                        AttributionConfig(lookback_s=a.lookback))
    _dump(res)


def cmd_live(a):
    import pandas as pd
    from .attribution import AttributionConfig, attribute_predicted_bottleneck_json
    from .collector import Collector, CollectorConfig
    from .model import LivePredictor
    pred = LivePredictor(a.model, a.interval, eval_every=a.eval_every)
    acfg = AttributionConfig(lookback_s=a.lookback, exclude_pids=(os.getpid(),))
    thr = a.threshold if a.threshold is not None else pred.b["threshold"]
    keep = int((acfg.lookback_s + 2) / a.interval) + 4
    dev_buf, proc_buf = deque(maxlen=keep), deque(maxlen=keep * 64)
    state = {"hits": 0, "cooldown_until": 0.0}

    def on_tick(ts, dev_rows, proc_rows):
        dev_buf.extend(dev_rows)
        proc_buf.extend(proc_rows)
        out = pred.push(dev_rows, proc_rows)
        if out is None:
            return
        t, p = out
        state["hits"] = state["hits"] + 1 if p >= thr else 0
        if a.verbose:
            print(f"{t:.2f} p={p:.3f}", flush=True)
        # Hysteresis: 2 consecutive positive evaluations (as in the offline
        # alarm_policy), then a cooldown, to avoid alert storms.
        if state["hits"] >= pred.b.get("alarm_consecutive", 2) and t >= state["cooldown_until"]:
            report = attribute_predicted_bottleneck_json(
                pd.DataFrame(list(proc_buf)), pd.DataFrame(list(dev_buf)), pred.device,
                t, p, thr, acfg)
            if report is not None:
                print(report, flush=True)
            state["cooldown_until"] = t + a.cooldown

    cfg = CollectorConfig(devices=[pred.device], interval_s=a.interval, proc_every=a.proc_every)
    meta = Collector(cfg, out_dir=None, on_tick=on_tick).run(a.duration)
    print(json.dumps({"overhead_incl_inference": meta["overhead"]}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="iobn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    home_work = os.path.expanduser("~/iobn-work")

    s = sp.add_parser("env"); s.add_argument("--workdir", default=os.path.expanduser("~")); s.set_defaults(f=cmd_env)
    s = sp.add_parser("wsl-check"); s.add_argument("--workdir", default=os.path.expanduser("~"))
    s.add_argument("--iostat-seconds", type=int, default=15); s.set_defaults(f=cmd_wsl_check)

    s = sp.add_parser("collect")
    s.add_argument("--out", required=True); s.add_argument("--duration", type=float)
    s.add_argument("--devices", nargs="*"); s.add_argument("--interval", type=float, default=0.25)
    s.add_argument("--proc-every", type=int, default=4); s.set_defaults(f=cmd_collect)

    s = sp.add_parser("experiment")
    s.add_argument("--scenario", default="all", choices=["burst", "mixed", "rogue", "all"])
    s.add_argument("--duration", type=float, default=300); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--workdir", default=home_work); s.add_argument("--out-root", default="runs")
    s.add_argument("--device"); s.add_argument("--data-mb", type=int, default=512)
    s.add_argument("--interval", type=float, default=0.25); s.add_argument("--proc-every", type=int, default=2)
    s.set_defaults(f=cmd_experiment)

    s = sp.add_parser("train")
    s.add_argument("runs", nargs="+"); s.add_argument("--device", required=True)
    s.add_argument("--out", default="model.joblib"); s.add_argument("--report")
    s.add_argument("--model", default="hgb", choices=["hgb", "rf"])
    s.add_argument("--h-start", type=float, default=1.5); s.add_argument("--h-end", type=float, default=3.0)
    s.add_argument("--test-frac", type=float, default=0.3)
    s.add_argument("--min-precision", type=float, default=0.80,
                   help="minimum out-of-fold precision for threshold selection (0-1)")
    s.add_argument("--alarm-consecutive", type=int, default=2,
                   help="consecutive evaluation scores required to raise an alarm")
    s.add_argument("--threshold-objective", choices=["recall", "f1"], default="recall",
                   help="maximize recall subject to the precision floor, or maximize F1")
    s.add_argument("--lat-floor-ms", type=float, default=2.0)
    s.add_argument("--lat-multiplier", type=float, default=3.0)
    s.add_argument("--min-iops", type=float, default=50.0)
    s.set_defaults(f=cmd_train)

    s = sp.add_parser("attribute-eval")
    s.add_argument("runs", nargs="+"); s.add_argument("--model", required=True)
    s.add_argument("--lookback", type=float, default=10.0); s.set_defaults(f=cmd_attribute_eval)

    s = sp.add_parser("live")
    s.add_argument("--model", required=True); s.add_argument("--duration", type=float)
    s.add_argument("--interval", type=float, default=0.25); s.add_argument("--proc-every", type=int, default=2)
    s.add_argument("--threshold", type=float); s.add_argument("--lookback", type=float, default=10.0)
    s.add_argument("--cooldown", type=float, default=10.0); s.add_argument("--verbose", action="store_true")
    s.add_argument("--eval-every", type=int, default=2, help="score every N ticks (overhead control)")
    s.set_defaults(f=cmd_live)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
