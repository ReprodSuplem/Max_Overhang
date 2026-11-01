#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maxOverhang_proposed.py
Main Dispatcher and Shared Solver Utilities for the Maximum Overhang Problem MIBLP Models.

Routes `--method` to the four progressive strengthening model modules:
  - M1_OrderPrune  -> ablation_m1_order_prune.py
  - M2_LocalMoment -> ablation_m2_local_moment.py
  - M3_ConvexHull  -> ablation_m3_convex_hull.py
  - M4_WarmStart   -> ablation_m4_warm_start.py
"""

import os
import sys
import time
import argparse
import gurobipy as gp
from gurobipy import GRB


class Logger(object):
    """Synchronize standard output to both the terminal and a target .out log file."""
    def __init__(self, filename="output.out"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class ProgressMonitorCallback:
    """Gurobi callback recording real-time primal and dual bound trajectories."""
    def __init__(self, start_time, log_interval=2.0):
        self.start_time = start_time
        self.log_interval = log_interval
        self.last_log_time = -1e9
        self.last_incumbent = -1e9
        self.first_sol_time = None

    def __call__(self, model, where):
        now = time.time() - self.start_time

        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            nodes = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
            if self.first_sol_time is None and obj > -1e8:
                self.first_sol_time = now
            gap = abs(bnd - obj) / max(1e-10, abs(obj)) * 100.0 if obj > -1e8 else float('inf')
            if abs(obj - self.last_incumbent) > 1e-6:
                self.last_incumbent = obj
                self.last_log_time = now
                print(f"[PROGRESS] Time: {now:8.2f}s | Event: NEW_SOL  | "
                      f"ObjVal: {obj:10.5f} | BestBound: {bnd:10.5f} | "
                      f"Gap: {gap:8.4f}% | Nodes: {int(nodes)}")

        elif where == GRB.Callback.MIP:
            if now - self.last_log_time >= self.log_interval:
                obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                nodes = model.cbGet(GRB.Callback.MIP_NODCNT)
                self.last_log_time = now
                if obj > -1e8 and abs(obj) < 1e10:
                    gap = abs(bnd - obj) / max(1e-10, abs(obj)) * 100.0
                    print(f"[PROGRESS] Time: {now:8.2f}s | Event: PERIODIC | "
                          f"ObjVal: {obj:10.5f} | BestBound: {bnd:10.5f} | "
                          f"Gap: {gap:8.4f}% | Nodes: {int(nodes)}")
                else:
                    print(f"[PROGRESS] Time: {now:8.2f}s | Event: PERIODIC | "
                          f"ObjVal:         NA | BestBound: {bnd:10.5f} | "
                          f"Gap:       NA | Nodes: {int(nodes)}")


def visualize_blocks(results, output_file, title_suffix="Proposed MIBLP", W_unit=0.5):
    """Render and save a 2D visualization of the balanced stack configuration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 7))
    block_height = 0.25

    table_rect = patches.Rectangle((-10, 0), 10, block_height,
                                   linewidth=1, edgecolor='black', facecolor='saddlebrown')
    ax.add_patch(table_rect)
    ax.text(-2, block_height / 2, 'Table (X <= 0)', color='white',
            ha='center', va='center', fontsize=12, fontweight='bold')

    for layer, pos_x, block_id, _ in results:
        y = layer * block_height
        x_left = pos_x - W_unit
        rect = patches.Rectangle((x_left, y), W_unit * 2, block_height,
                                 linewidth=1.5, edgecolor='black', facecolor='lightcoral', alpha=0.88)
        ax.add_patch(rect)
        ax.text(pos_x, y + block_height / 2, f'B{block_id}\n({pos_x:.3f})',
                ha='center', va='center', fontsize=8)

    all_x = [pos_x for _, pos_x, _, _ in results]
    min_x = min([-2] + all_x) - 2.0
    max_overhang_val = max([pos_x + W_unit for _, pos_x, _, _ in results])
    max_x = max([1] + all_x) + 3.0
    max_layer = max([layer for layer, _, _, _ in results])

    ax.set_xlim(min_x, max_x)
    y_max = (max_layer + 2) * block_height
    ax.set_ylim(0, y_max)
    ax.set_aspect('equal')

    ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, label='Table Edge (X=0)')
    ax.axvline(x=max_overhang_val, color='#FF9900', linestyle='-.', alpha=0.9, label='Max Overhang')
    ax.text(max_overhang_val + 0.05, y_max / 2.0, f'{max_overhang_val:.5f}',
            color='#FF9900', va='center', ha='left', fontsize=12, fontweight='bold')

    plt.title(f'Overhang Stack Configuration - {title_suffix} (N={len(results)})')
    plt.xlabel('Horizontal Coordinate X')
    plt.ylabel('Layer Index')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def execute_and_report(model, X, L, T, N, method_name, time_limit_sec, random_seed, threads, mip_gap, log_interval=2.0, eta=0.0):
    """Configure Gurobi solver parameters, execute optimization, and print the standardized summary report."""
    model.setParam("NonConvex", 2)
    model.setParam("Threads", threads)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Seed", random_seed)
    if time_limit_sec > 0:
        model.setParam("TimeLimit", float(time_limit_sec))

    print(f"\n================================================================================")
    print(f"META: Method={method_name} | N={N} | Eta={eta:.4f} | Seed={random_seed} | TimeLimit={time_limit_sec}s | "
          f"Threads={threads} | MIPGap={mip_gap}")
    print(f"================================================================================")

    start_time = time.time()
    monitor = ProgressMonitorCallback(start_time, log_interval=log_interval)

    try:
        model.optimize(monitor)
    except KeyboardInterrupt:
        print("\n[!] Solve interrupted by user (KeyboardInterrupt)!")

    duration = time.time() - start_time

    if model.Status == GRB.OPTIMAL:
        status_str = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        status_str = "TIMEOUT"
    elif model.Status == GRB.INTERRUPTED:
        status_str = "INTERRUPTED"
    elif model.Status == GRB.INFEASIBLE:
        status_str = "INFEASIBLE"
    else:
        status_str = f"STATUS_{model.Status}"

    obj_val = model.ObjVal if model.SolCount > 0 else float('nan')
    best_bound = model.ObjBound if hasattr(model, "ObjBound") else float('nan')
    mip_gap_pct = (model.MIPGap * 100.0) if (model.SolCount > 0 and hasattr(model, "MIPGap")) else float('nan')
    node_count = int(model.NodeCount) if hasattr(model, "NodeCount") else 0
    first_sol_t = monitor.first_sol_time if monitor.first_sol_time is not None else float('nan')

    print(f"\n================================================================================")
    print(f"SUMMARY_REPORT:")
    print(f"  Method           : {method_name}")
    print(f"  N                : {N}")
    print(f"  Eta              : {eta:.4f}")
    print(f"  Status           : {status_str}")
    print(f"  ObjVal           : {obj_val:.5f}")
    print(f"  BestBound        : {best_bound:.5f}")
    print(f"  MIPGap_Pct       : {mip_gap_pct:.4f}%")
    print(f"  Runtime_Sec      : {duration:.4f}")
    print(f"  FirstSol_Sec     : {first_sol_t:.4f}")
    print(f"  NodeCount        : {node_count}")
    print(f"  SolCount         : {model.SolCount}")
    print(f"================================================================================")

    if model.SolCount > 0:
        results = []
        for i in range(1, N + 1):
            layer = int(round(L[i].X))
            pos = X[i].X
            if i == 1:
                supports = [0]
            else:
                supports = [j for j in range(1, i) if (i, j) in T and T[i, j].X > 0.5]
            results.append((layer, pos, i, supports))
        results.sort(key=lambda x: (-x[0], x[1]))

        print("\nSTACK_CONFIGURATION:")
        for layer, pos, i, supports in results:
            print(f"  Layer {layer:2d} | Block {i:2d} | Center X: {pos:8.5f} | Supported by: {supports}")
        return results
    return None


def dispatch_solver(N, method, time_limit_sec, random_seed, threads=1, mip_gap=5e-4, eta=0.0):
    """Dispatch execution to the selected model variant module."""
    if method == "M1_OrderPrune":
        from ablation_m1_order_prune import solve_m1_order_prune
        return solve_m1_order_prune(N, time_limit_sec, random_seed, threads, mip_gap, method_label=method, eta=eta)
    elif method == "M2_LocalMoment":
        from ablation_m2_local_moment import solve_m2_local_moment
        return solve_m2_local_moment(N, time_limit_sec, random_seed, threads, mip_gap, method_label=method, eta=eta)
    elif method == "M3_ConvexHull":
        from ablation_m3_convex_hull import solve_m3_convex_hull
        return solve_m3_convex_hull(N, time_limit_sec, random_seed, threads, mip_gap, eta=eta)
    elif method == "M4_WarmStart":
        from ablation_m4_warm_start import solve_m4_warm_start
        return solve_m4_warm_start(N, time_limit_sec, random_seed, threads, mip_gap, eta=eta)
    else:
        raise ValueError(f"Unknown model variant: {method}")


def main():
    parser = argparse.ArgumentParser(description="Proposed MIBLP Solver (M1-M4) for the Maximum Overhang Problem")
    parser.add_argument("N", type=int, help="Number of blocks (e.g., 10)")
    parser.add_argument("time_limit", type=int, help="Time limit in seconds (0 = no limit)")
    parser.add_argument("--method", choices=["M1_OrderPrune", "M2_LocalMoment", "M3_ConvexHull", "M4_WarmStart"],
                        default="M4_WarmStart", help="Model variant to solve (default: M4_WarmStart)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--threads", "-t", type=int, default=1, help="Number of solver threads (default: 1)")
    parser.add_argument("--gap", type=float, default=5e-4, help="Relative MIPGap tolerance (default: 5e-4)")
    parser.add_argument("--eta", type=float, default=0.0, help="Relative stability margin eta = epsilon / W (default: 0.0)")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for .out and .png files")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate and save stack visualization (.png) inside outdir (default: False)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    eta_suffix = f"_eta{args.eta:.2f}" if args.eta > 0.0 else ""
    base_name = os.path.join(args.outdir, f"{args.method}_N{args.N}{eta_suffix}")
    out_file = f"{base_name}.out"

    logger = Logger(out_file)
    sys.stdout = logger

    try:
        print(f"--- EXPERIMENT START | METHOD = {args.method} | N = {args.N} | ETA = {args.eta:.4f} | SEED = {args.seed} ---")
        results = dispatch_solver(
            N=args.N,
            method=args.method,
            time_limit_sec=args.time_limit,
            random_seed=args.seed,
            threads=args.threads,
            mip_gap=args.gap,
            eta=args.eta
        )
        if results and args.plot:
            png_file = f"{base_name}.png"
            visualize_blocks(results, png_file, title_suffix=f"{args.method} (eta={args.eta:.2f})")
            print(f"INFO: Saved visualization to {png_file}")
        print(f"--- EXPERIMENT END | METHOD = {args.method} | N = {args.N} | ETA = {args.eta:.4f} ---")
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
