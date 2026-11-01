#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py
Parallel Batch Runner for the Maximum Overhang MIBLP Ablation Study (Models M0 through M4).

All `.out` logs and `ablation_summary.csv` are stored inside `--outdir`.
Generating `convergence_trace.csv` is optional (`--save-trace`) and disabled by default.
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from parse_results import parse_single_out_file


ABLATION_METHODS = [
    "Baseline",
    "M1_OrderPrune",
    "M2_LocalMoment",
    "M3_ConvexHull",
    "M4_WarmStart",
]


def fmt_time_now():
    return datetime.now().strftime("%H:%M:%S")


def run_single_job(job_info, active_procs, lock):
    """Execute a single solver job as a subprocess and return parsed summary metrics."""
    idx = job_info["idx"]
    total = job_info["total"]
    N = job_info["N"]
    method = job_info["method"]
    cmd = job_info["cmd"]
    out_file = job_info["out_file"]
    parallel_mode = job_info["parallel_mode"]
    threads = job_info["threads"]

    start_ts = time.time()
    with lock:
        print(f"[{fmt_time_now()}] [START {idx:02d}/{total:02d}] "
              f"Method={method:<14} | N={N:<2d} | Threads={threads} -> {out_file}",
              flush=True)

    proc = None
    try:
        if parallel_mode:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        else:
            proc = subprocess.Popen(cmd, start_new_session=True)

        with lock:
            active_procs.add(proc)

        proc.wait()
    finally:
        if proc is not None:
            with lock:
                active_procs.discard(proc)

    wall_sec = time.time() - start_ts

    rec = None
    if os.path.exists(out_file):
        try:
            rec, _ = parse_single_out_file(out_file, collect_trace=False)
        except Exception:
            rec = None

    return {
        "idx": idx,
        "total": total,
        "N": N,
        "method": method,
        "out_file": out_file,
        "wall_sec": wall_sec,
        "record": rec,
    }


def update_csv_reports(outdir, save_trace=False):
    """Incrementally update ablation_summary.csv (and optionally convergence_trace.csv) inside outdir."""
    if os.path.exists("parse_results.py"):
        csv_path = os.path.join(outdir, "ablation_summary.csv")
        cmd = [
            sys.executable, "parse_results.py",
            "--indir", outdir,
            "--output", csv_path,
        ]
        if save_trace:
            trace_path = os.path.join(outdir, "convergence_trace.csv")
            cmd.extend(["--save-trace", "--trace-output", trace_path])
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Batch Runner for Maximum Overhang MIBLP Ablation Study"
    )
    parser.add_argument("--n-list", type=int, nargs="+", default=list(range(6, 36)),
                        help="List of block counts N to evaluate (default: 6..35)")
    parser.add_argument("--methods", type=str, nargs="+", choices=ABLATION_METHODS, default=ABLATION_METHODS,
                        help="Model variants to evaluate (default: all 5 models)")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Time limit per run in seconds (default: 3600)")
    parser.add_argument("--workers", "-j", "--parallel", dest="workers", type=int, default=5,
                        help="Number of concurrent solver processes (default: 5)")
    parser.add_argument("--threads", "-t", dest="threads", type=int, default=1,
                        help="Number of Gurobi threads per solver process (default: 1)")
    parser.add_argument("--gap", type=float, default=5e-4,
                        help="Relative MIPGap tolerance (default: 5e-4 = 0.05%%)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="Relative stability margin eta = epsilon / W (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for solver reproducibility (default: 42)")
    parser.add_argument("--outdir", type=str, default="results_N6_35",
                        help="Output directory for .out logs and ablation_summary.csv (default: results_N6_35)")
    parser.add_argument("--save-trace", action="store_true", default=False,
                        help="Optionally save convergence_trace.csv inside outdir for debugging (default: False)")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Optionally save .png stack visualizations inside outdir (default: False)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    jobs = []
    total_runs = len(args.n_list) * len(args.methods)
    run_idx = 0
    eta_suffix = f"_eta{args.eta:.2f}" if args.eta > 0.0 else ""

    for N in args.n_list:
        for method in args.methods:
            run_idx += 1
            out_file = os.path.join(args.outdir, f"{method}_N{N}{eta_suffix}.out")
            if method == "Baseline":
                cmd = [
                    sys.executable, "maxOverhang_baseline.py",
                    str(N), str(args.time_limit),
                    "--seed", str(args.seed),
                    "--threads", str(args.threads),
                    "--gap", str(args.gap),
                    "--eta", str(args.eta),
                    "--outdir", args.outdir
                ]
            else:
                cmd = [
                    sys.executable, "maxOverhang_proposed.py",
                    str(N), str(args.time_limit),
                    "--method", method,
                    "--seed", str(args.seed),
                    "--threads", str(args.threads),
                    "--gap", str(args.gap),
                    "--eta", str(args.eta),
                    "--outdir", args.outdir
                ]
            if args.plot:
                cmd.append("--plot")

            jobs.append({
                "idx": run_idx,
                "total": total_runs,
                "N": N,
                "method": method,
                "cmd": cmd,
                "out_file": out_file,
                "parallel_mode": (args.workers > 1),
                "threads": args.threads,
            })

    total_cpu_threads = args.workers * args.threads
    print("================================================================================")
    print(f"Starting Maximum Overhang MIBLP Ablation Suite ({total_runs} total runs)")
    print(f"  N list                      : {args.n_list}")
    print(f"  Methods                     : {args.methods}")
    print(f"  Time Limit per Run          : {args.time_limit} s")
    print(f"  MIPGap Tolerance (--gap)    : {args.gap} ({args.gap * 100:.3f}%)")
    print(f"  Stability Margin (--eta)    : {args.eta}")
    print(f"  Parallel Workers (-j)       : {args.workers} concurrent processes")
    print(f"  Gurobi Threads/Worker (-t)  : {args.threads} threads per process")
    print(f"  Total CPU Threads Utilized  : {total_cpu_threads} threads ({args.workers} x {args.threads})")
    print(f"  Output Directory (--outdir) : {args.outdir}/")
    print(f"  Save Trace (--save-trace)   : {args.save_trace}")
    print(f"  Plot Figures (--plot)       : {args.plot}")
    print("================================================================================", flush=True)

    lock = threading.Lock()
    active_procs = set()
    suite_start = time.time()
    completed_count = 0

    try:
        if args.workers <= 1:
            for job in jobs:
                res = run_single_job(job, active_procs, lock)
                completed_count += 1
                rec = res["record"]
                if rec:
                    obj_s = f"{rec['ObjVal']:.5f}" if rec["ObjVal"] is not None else "NA"
                    bnd_s = f"{rec['BestBound']:.5f}" if rec["BestBound"] is not None else "NA"
                    gap_s = f"{rec['MIPGap_Pct']:.4f}%" if rec["MIPGap_Pct"] is not None else "NA"
                    nds_s = str(rec["NodeCount"]) if rec["NodeCount"] is not None else "NA"
                    print(f"[{fmt_time_now()}] [DONE  {completed_count:02d}/{total_runs:02d}] "
                          f"Method={res['method']:<14} | N={res['N']:<2d} | Status={rec['Status']:<8} | "
                          f"ObjVal={obj_s} | Bound={bnd_s} | Gap={gap_s} | "
                          f"Time={res['wall_sec']:.2f}s | Nodes={nds_s}", flush=True)
                update_csv_reports(args.outdir, save_trace=args.save_trace)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {
                    executor.submit(run_single_job, job, active_procs, lock): job
                    for job in jobs
                }
                for future in as_completed(future_to_job):
                    res = future.result()
                    with lock:
                        completed_count += 1
                        rec = res["record"]
                        if rec:
                            obj_s = f"{rec['ObjVal']:.5f}" if rec["ObjVal"] is not None else "NA"
                            bnd_s = f"{rec['BestBound']:.5f}" if rec["BestBound"] is not None else "NA"
                            gap_s = f"{rec['MIPGap_Pct']:.4f}%" if rec["MIPGap_Pct"] is not None else "NA"
                            nds_s = str(rec["NodeCount"]) if rec["NodeCount"] is not None else "NA"
                            print(f"[{fmt_time_now()}] [DONE  {completed_count:02d}/{total_runs:02d}] "
                                  f"Method={res['method']:<14} | N={res['N']:<2d} | Status={rec['Status']:<8} | "
                                  f"ObjVal={obj_s} | Bound={bnd_s} | Gap={gap_s} | "
                                  f"Time={res['wall_sec']:.2f}s | Nodes={nds_s}", flush=True)
                        else:
                            print(f"[{fmt_time_now()}] [DONE  {completed_count:02d}/{total_runs:02d}] "
                                  f"Method={res['method']:<14} | N={res['N']:<2d} | Time={res['wall_sec']:.2f}s",
                                  flush=True)
                        update_csv_reports(args.outdir, save_trace=args.save_trace)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected! Terminating all active Gurobi processes...", flush=True)
        with lock:
            for p in list(active_procs):
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    try:
                        p.terminate()
                    except Exception:
                        pass
        sys.exit(1)

    total_wall = time.time() - suite_start
    csv_path = os.path.join(args.outdir, "ablation_summary.csv")
    print("\n================================================================================")
    print(f"All {completed_count}/{total_runs} runs finished in {total_wall:.2f}s wall-clock time.")
    print(f"Final statistical summary saved to: {csv_path}")
    print("================================================================================")
    if os.path.exists("parse_results.py"):
        cmd = [
            sys.executable, "parse_results.py",
            "--indir", args.outdir,
            "--output", csv_path,
        ]
        if args.save_trace:
            trace_path = os.path.join(args.outdir, "convergence_trace.csv")
            cmd.extend(["--save-trace", "--trace-output", trace_path])
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
