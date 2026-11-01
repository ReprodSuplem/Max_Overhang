#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maxOverhang_baseline.py
Baseline Mixed-Integer Bilinear Programming (MIBLP) Formulation (M0)
for the Maximum Overhang Problem over Directed Acyclic Graph (DAG) Support Networks.
"""

import os
import sys
import time
import argparse
import gurobipy as gp
from gurobipy import GRB
from maxOverhang_proposed import Logger, ProgressMonitorCallback, visualize_blocks


def solve_overhang_tower_baseline(N, time_limit_sec, random_seed, threads=1, mip_gap=5e-4, log_interval=2.0, eta=0.0):
    """Build and solve the baseline MIBLP formulation (M0) for an N-block overhang stack."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model("TowerOfLire_Baseline_DAG", env=env)

    W = 0.5
    MASS = 1.0
    EPSILON = W * float(eta)
    MAX_X = float(N)
    MAX_F = float(N) * MASS
    MAX_TORQUE = MAX_F * MAX_X
    BIG_M = MAX_X * 2.0 + 1.0

    # 1. Decision Variables
    X = {}
    X[0] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name="X_0")
    X[1] = model.addVar(lb=-W + EPSILON, ub=W - 2.0 * EPSILON, vtype=GRB.CONTINUOUS, name="X_1")
    for i in range(2, N + 1):
        X[i] = model.addVar(lb=-MAX_X, ub=MAX_X - W, vtype=GRB.CONTINUOUS, name=f"X_{i}")

    L = {}
    L[0] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name="L_0")
    L[1] = model.addVar(lb=1, ub=1, vtype=GRB.INTEGER, name="L_1")
    for i in range(2, N + 1):
        L[i] = model.addVar(lb=2, ub=N, vtype=GRB.INTEGER, name=f"L_{i}")

    T = {}; F = {}; P = {}; Torque = {}
    for i in range(1, N + 1):
        for j in range(0, N + 1):
            if i == j:
                T[i, j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY, name=f"T_{i}_{j}")
                F[i, j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"F_{i}_{j}")
                P[i, j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"P_{i}_{j}")
                Torque[i, j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"Torque_{i}_{j}")
            elif j == 0:
                t_val = 1.0 if i == 1 else 0.0
                T[i, 0] = model.addVar(lb=t_val, ub=t_val, vtype=GRB.BINARY, name=f"T_{i}_0")
                F[i, 0] = model.addVar(lb=0.0, ub=MAX_F * t_val, vtype=GRB.CONTINUOUS, name=f"F_{i}_0")
                P[i, 0] = model.addVar(lb=-MAX_X * t_val, ub=MAX_X * t_val, vtype=GRB.CONTINUOUS, name=f"P_{i}_0")
                Torque[i, 0] = model.addVar(lb=-MAX_TORQUE * t_val, ub=MAX_TORQUE * t_val,
                                            vtype=GRB.CONTINUOUS, name=f"Torque_{i}_0")
            else:
                T[i, j] = model.addVar(vtype=GRB.BINARY, name=f"T_{i}_{j}")
                F[i, j] = model.addVar(lb=0.0, ub=MAX_F, vtype=GRB.CONTINUOUS, name=f"F_{i}_{j}")
                P[i, j] = model.addVar(lb=-MAX_X, ub=MAX_X, vtype=GRB.CONTINUOUS, name=f"P_{i}_{j}")
                Torque[i, j] = model.addVar(lb=-MAX_TORQUE, ub=MAX_TORQUE,
                                            vtype=GRB.CONTINUOUS, name=f"Torque_{i}_{j}")

    # 2. Geometric Contact and DAG Topological Constraints
    for i in range(1, N + 1):
        model.addConstr(gp.quicksum(T[i, j] for j in range(0, N + 1) if i != j) >= 1, name=f"DAGSup_{i}")
        for j in range(0, N + 1):
            if i == j:
                continue
            model.addConstr(F[i, j] <= MAX_F * T[i, j])
            model.addConstr(P[i, j] >= -MAX_X * T[i, j])
            model.addConstr(P[i, j] <= MAX_X * T[i, j])
            model.addConstr(Torque[i, j] >= -MAX_TORQUE * T[i, j])
            model.addConstr(Torque[i, j] <= MAX_TORQUE * T[i, j])

            model.addConstr(P[i, j] >= X[i] - W + EPSILON - BIG_M * (1 - T[i, j]))
            model.addConstr(P[i, j] <= X[i] + W - EPSILON + BIG_M * (1 - T[i, j]))
            if j == 0:
                model.addConstr(P[i, 0] <= -EPSILON + BIG_M * (1 - T[i, 0]))
            else:
                model.addConstr(P[i, j] >= X[j] - W + EPSILON - BIG_M * (1 - T[i, j]))
                model.addConstr(P[i, j] <= X[j] + W - EPSILON + BIG_M * (1 - T[i, j]))

            model.addConstr(L[i] - L[j] <= 1 + N * (1 - T[i, j]))
            model.addConstr(L[i] - L[j] >= 1 - N * (1 - T[i, j]))
            model.addConstr(Torque[i, j] == F[i, j] * P[i, j], name=f"Bilinear_{i}_{j}")

    # 3. Same-Layer Non-Overlap and Layer Ordering Constraints
    for i in range(1, N):
        model.addConstr(L[i] <= L[i + 1], name=f"SymLayer_{i}")
        b = model.addVar(vtype=GRB.BINARY, name=f"LayerJump_{i}")
        model.addConstr(L[i + 1] - L[i] >= b)
        model.addConstr(L[i + 1] - L[i] <= N * b)
        model.addConstr(X[i] - X[i + 1] >= 2 * W - BIG_M * b, name=f"NonOverlap_{i}")

    # 4. Static Force and Moment Equilibrium Equations
    model.addConstr(gp.quicksum(X[i] for i in range(1, N + 1)) <= -float(N) * EPSILON, name="Global_CoM")
    for i in range(1, N + 1):
        f_in = gp.quicksum(F[k, i] for k in range(1, N + 1) if k != i) + MASS
        f_out = gp.quicksum(F[i, j] for j in range(0, N + 1) if j != i)
        model.addConstr(f_in == f_out, name=f"ForceBal_{i}")

        t_in = gp.quicksum(Torque[k, i] for k in range(1, N + 1) if k != i) + MASS * X[i]
        t_out = gp.quicksum(Torque[i, j] for j in range(0, N + 1) if j != i)
        model.addConstr(t_in == t_out, name=f"TorqueBal_{i}")

    # 5. Objective Function
    right_edges = [model.addVar(lb=-MAX_X, ub=MAX_X, vtype=GRB.CONTINUOUS, name=f"RE_{i}") for i in range(1, N + 1)]
    for i in range(1, N + 1):
        model.addConstr(right_edges[i - 1] == X[i] + W)

    max_overhang = model.addVar(lb=0.5 - 2.0 * EPSILON, ub=MAX_X, vtype=GRB.CONTINUOUS, name="max_overhang")
    model.addGenConstrMax(max_overhang, right_edges, name="LinkMaxEdge")
    model.setObjective(max_overhang, GRB.MAXIMIZE)
    model.update()

    model.setParam("NonConvex", 2)
    model.setParam("Threads", threads)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Seed", random_seed)
    if time_limit_sec > 0:
        model.setParam("TimeLimit", float(time_limit_sec))

    print(f"\n================================================================================")
    print(f"META: Method=Baseline | N={N} | Eta={eta:.4f} | Seed={random_seed} | TimeLimit={time_limit_sec}s | "
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
    print(f"  Method           : Baseline")
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
            supports = [j for j in range(0, N + 1) if i != j and T[i, j].X > 0.5]
            results.append((layer, pos, i, supports))
        results.sort(key=lambda x: (-x[0], x[1]))

        print("\nSTACK_CONFIGURATION:")
        for layer, pos, i, supports in results:
            print(f"  Layer {layer:2d} | Block {i:2d} | Center X: {pos:8.5f} | Supported by: {supports}")
        return results
    return None


def main():
    parser = argparse.ArgumentParser(description="Baseline MIBLP Solver (M0) for the Maximum Overhang Problem")
    parser.add_argument("N", type=int, help="Number of blocks (e.g., 10)")
    parser.add_argument("time_limit", type=int, help="Time limit in seconds (0 = no limit)")
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
    base_name = os.path.join(args.outdir, f"Baseline_N{args.N}{eta_suffix}")
    out_file = f"{base_name}.out"

    logger = Logger(out_file)
    sys.stdout = logger

    try:
        print(f"--- EXPERIMENT START | METHOD = Baseline | N = {args.N} | ETA = {args.eta:.4f} | SEED = {args.seed} ---")
        results = solve_overhang_tower_baseline(
            N=args.N,
            time_limit_sec=args.time_limit,
            random_seed=args.seed,
            threads=args.threads,
            mip_gap=args.gap,
            eta=args.eta
        )
        if results and args.plot:
            png_file = f"{base_name}.png"
            visualize_blocks(results, png_file, title_suffix=f"Baseline (eta={args.eta:.2f})")
            print(f"INFO: Saved visualization to {png_file}")
        print(f"--- EXPERIMENT END | METHOD = Baseline | N = {args.N} ---")
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
