#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_m2_local_moment.py
Model Variant M2 (M2_LocalMoment):
  - Incorporates Model M1 topological reduction and base reaction linearization
  - Recursive sub-stack overhang upper bound D^*(N - 1)
  - Block-indexed coordinate reachability domain tightening [x_lb[i], x_ub[i]]
  - Disjunctive contact-interval hull and global table-torque identity
"""

import math
import gurobipy as gp
from gurobipy import GRB
from maxOverhang_proposed import execute_and_report


def _solve_loaded_spine_max_single(m, eta=0.0):
    """Compute exact maximum displacement sum S^*(m, eta) of a loaded spinal stack of weight m."""
    w_eff = 0.5 * (1.0 - float(eta))
    L_eff = 2.0 * w_eff
    if m <= 0:
        return 0.0
    if m == 1:
        return w_eff
    best_s = w_eff
    for k in range(1, m + 1):
        def eval_t1(t1):
            t = [0.0] * (k + 1)
            w = [0.0] * (k + 1)
            t[0] = 0.0
            t[1] = t1
            w[1] = t1 - 1.0
            for idx in range(1, k):
                t_next = (t[idx] ** 2) / (t[idx - 1] + 0.5)
                if t_next < t[idx] + 1.0:
                    t_next = t[idx] + 1.0
                t[idx + 1] = t_next
                w[idx + 1] = t[idx + 1] - t[idx] - 1.0
            return t, w

        lo, hi = 1.0, float(m)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            t, w = eval_t1(mid)
            if t[k] < float(m):
                lo = mid
            else:
                hi = mid
        t, w = eval_t1(0.5 * (lo + hi))
        d = [0.0] + [L_eff * (w[idx] + 0.5) / t[idx] for idx in range(1, k + 1)]
        best_s = max(best_s, sum(d[1:]))
    return best_s


def _compute_recursive_substack_ub(m, eta=0.0):
    """
    Compute a valid upper bound D^*(m, eta) on the overhang of an m-block sub-stack:
      - For m <= 19: exact loaded spinal maximum S^*(m, eta)
      - For m >= 20: recursive sub-stack induction starting from D^*(19, eta) = S^*(19, eta)
    """
    w_eff = 0.5 * (1.0 - float(eta))
    L_eff = 2.0 * w_eff
    if m <= 0:
        return 0.0
    if m == 1:
        return w_eff
    if m <= 19:
        return _solve_loaded_spine_max_single(m, eta=eta) + 1e-4

    val = _solve_loaded_spine_max_single(19, eta=eta) + 1e-4
    for r in range(20, m + 1):
        val += L_eff * (1.0 - 1.0 / (2.0 * float(r)))
    return min(val, float(m) * w_eff + 0.5, 4.5 * math.pow(float(m), 1.0 / 3.0) + 1.0)


def solve_m2_local_moment(N, time_limit_sec, random_seed, threads=1, mip_gap=5e-4, method_label="M2_LocalMoment", eta=0.0):
    """Build and solve Model M2 for an N-block overhang stack."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model(f"TowerOfLire_{method_label}", env=env)

    W = 0.5
    MASS = 1.0
    EPSILON = W * float(eta)
    w_eff = W - EPSILON

    base_ub = min(float(N) * w_eff + W, 4.5 * math.pow(N, 1.0 / 3.0) + 1.0)
    if N >= 14:
        sub_stack_ub = _compute_recursive_substack_ub(N - 1, eta=eta)
        theo_overhang_ub = min(base_ub, (float(N - 1) / float(N)) * w_eff - EPSILON + W + sub_stack_ub)
    else:
        theo_overhang_ub = base_ub

    MAX_X = max(2.0, min(float(N), theo_overhang_ub + 0.5))

    x_lb = {0: 0.0, 1: -W + EPSILON}
    x_ub = {0: 0.0, 1: W - 2.0 * EPSILON}
    for i in range(2, N + 1):
        x_lb[i] = -MAX_X
        x_ub[i] = min(MAX_X - W, x_ub[1] + (i - 1) * (2.0 * w_eff), theo_overhang_ub - W)

    # 1. Decision Variables
    X = {}
    X[0] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name="X_0")
    for i in range(1, N + 1):
        X[i] = model.addVar(lb=x_lb[i], ub=x_ub[i], vtype=GRB.CONTINUOUS, name=f"X_{i}")

    L = {}
    L[0] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name="L_0")
    L[1] = model.addVar(lb=1, ub=1, vtype=GRB.INTEGER, name="L_1")
    for i in range(2, N + 1):
        L[i] = model.addVar(lb=2, ub=i, vtype=GRB.INTEGER, name=f"L_{i}")

    T = {}; F = {}; P = {}; Torque = {}; max_f_i = {}; p_lb = {}; p_ub = {}

    # Unique Table Base (i = 1, j = 0): F_{1,0} == N * MASS is constant
    max_f_i[1] = float(N) * MASS
    p_lb[1, 0] = -2.0 * W + EPSILON
    p_ub[1, 0] = -EPSILON
    T[1, 0] = model.addVar(lb=1.0, ub=1.0, vtype=GRB.BINARY, name="T_1_0")
    F[1, 0] = model.addVar(lb=max_f_i[1], ub=max_f_i[1], vtype=GRB.CONTINUOUS, name="F_1_0")
    P[1, 0] = model.addVar(lb=p_lb[1, 0], ub=p_ub[1, 0], vtype=GRB.CONTINUOUS, name="P_1_0")
    Torque[1, 0] = model.addVar(lb=max_f_i[1] * p_lb[1, 0], ub=max_f_i[1] * p_ub[1, 0], vtype=GRB.CONTINUOUS, name="Torque_1_0")

    model.addConstr(P[1, 0] >= X[1] - w_eff, name="BaseContact_LB")
    model.addConstr(P[1, 0] <= X[1] + w_eff, name="BaseContact_UB")
    model.addConstr(Torque[1, 0] == max_f_i[1] * P[1, 0], name="BaseTorque_Linear")

    for i in range(2, N + 1):
        max_f_i[i] = float(N - i + 1) * MASS
        for j in range(1, i):
            u_f = max_f_i[i]
            p_lb[i, j] = max(x_lb[i] - w_eff, x_lb[j] - w_eff)
            p_ub[i, j] = min(x_ub[i] + w_eff, x_ub[j] + w_eff)
            t_lb = min(0.0, u_f * p_lb[i, j])
            t_ub = max(0.0, u_f * p_ub[i, j])

            T[i, j] = model.addVar(vtype=GRB.BINARY, name=f"T_{i}_{j}")
            F[i, j] = model.addVar(lb=0.0, ub=u_f, vtype=GRB.CONTINUOUS, name=f"F_{i}_{j}")
            P[i, j] = model.addVar(lb=min(0.0, p_lb[i, j]), ub=max(0.0, p_ub[i, j]),
                                   vtype=GRB.CONTINUOUS, name=f"P_{i}_{j}")
            Torque[i, j] = model.addVar(lb=t_lb, ub=t_ub,
                                        vtype=GRB.CONTINUOUS, name=f"Torque_{i}_{j}")

    # 2. Disjunctive Contact-Interval Hull and Topological Constraints (for i >= 2)
    for i in range(2, N + 1):
        u_f = max_f_i[i]
        model.addConstr(gp.quicksum(T[i, j] for j in range(1, i)) >= 1, name=f"DAGSupMin_{i}")
        model.addConstr(gp.quicksum(T[i, j] for j in range(1, i)) <= 2, name=f"DAGSupMax_{i}")

        for j in range(1, i):
            l_p = p_lb[i, j]
            u_p = p_ub[i, j]

            model.addConstr(F[i, j] <= u_f * T[i, j], name=f"ForceCap_{i}_{j}")
            model.addConstr(P[i, j] >= l_p * T[i, j], name=f"PCapLB_{i}_{j}")
            model.addConstr(P[i, j] <= u_p * T[i, j], name=f"PCapUB_{i}_{j}")
            model.addConstr(Torque[i, j] >= min(0.0, u_f * l_p) * T[i, j], name=f"TCapLB_{i}_{j}")
            model.addConstr(Torque[i, j] <= max(0.0, u_f * u_p) * T[i, j], name=f"TCapUB_{i}_{j}")

            model.addConstr(P[i, j] - X[i] >= -w_eff * T[i, j] - x_ub[i] * (1 - T[i, j]),
                            name=f"PerspInt_iLB_{i}_{j}")
            model.addConstr(P[i, j] - X[i] <= w_eff * T[i, j] - x_lb[i] * (1 - T[i, j]),
                            name=f"PerspInt_iUB_{i}_{j}")
            model.addConstr(P[i, j] - X[j] >= -w_eff * T[i, j] - x_ub[j] * (1 - T[i, j]),
                            name=f"PerspInt_jLB_{i}_{j}")
            model.addConstr(P[i, j] - X[j] <= w_eff * T[i, j] - x_lb[j] * (1 - T[i, j]),
                            name=f"PerspInt_jUB_{i}_{j}")

            model.addConstr(L[i] - L[j] <= 1 + (i - 1) * (1 - T[i, j]), name=f"LayerUB_{i}_{j}")
            model.addConstr(L[i] - L[j] >= 1 - j * (1 - T[i, j]), name=f"LayerLB_{i}_{j}")
            model.addConstr(Torque[i, j] == F[i, j] * P[i, j], name=f"Bilinear_{i}_{j}")

    # 3. Same-Layer Non-Overlap and Layer Ordering Constraints
    for i in range(1, N):
        model.addConstr(L[i] <= L[i + 1], name=f"SymLayer_{i}")
        b = model.addVar(vtype=GRB.BINARY, name=f"LayerJump_{i}")
        model.addConstr(L[i + 1] - L[i] >= b)
        model.addConstr(L[i + 1] - L[i] <= i * b)
        span_m = (x_ub[i + 1] - x_lb[i]) + 2.0 * W
        model.addConstr(X[i] - X[i + 1] >= 2 * W - span_m * b, name=f"NonOverlap_{i}")

    # 4. Static Force and Moment Equilibrium Equations with Global Table-Torque Identity
    model.addConstr(gp.quicksum(X[i] for i in range(1, N + 1)) <= -float(N) * EPSILON, name="Global_CoM")
    model.addConstr(gp.quicksum(MASS * X[i] for i in range(1, N + 1)) == Torque[1, 0],
                    name="Global_Table_Torque_Identity")

    model.addConstr(gp.quicksum(F[k, 1] for k in range(2, N + 1)) + MASS == F[1, 0], name="ForceBal_1")
    model.addConstr(gp.quicksum(Torque[k, 1] for k in range(2, N + 1)) + MASS * X[1] == Torque[1, 0], name="TorqueBal_1")

    for i in range(2, N + 1):
        f_in = gp.quicksum(F[k, i] for k in range(i + 1, N + 1)) + MASS
        f_out = gp.quicksum(F[i, j] for j in range(1, i))
        model.addConstr(f_in == f_out, name=f"ForceBal_{i}")

        t_in = gp.quicksum(Torque[k, i] for k in range(i + 1, N + 1)) + MASS * X[i]
        t_out = gp.quicksum(Torque[i, j] for j in range(1, i))
        model.addConstr(t_in == t_out, name=f"TorqueBal_{i}")

    # 5. Objective Function
    right_edges = [model.addVar(lb=x_lb[i] + W, ub=x_ub[i] + W, vtype=GRB.CONTINUOUS, name=f"RE_{i}")
                   for i in range(1, N + 1)]
    for i in range(1, N + 1):
        model.addConstr(right_edges[i - 1] == X[i] + W)

    max_overhang = model.addVar(lb=0.5 - 2.0 * EPSILON, ub=theo_overhang_ub, vtype=GRB.CONTINUOUS, name="max_overhang")
    model.addGenConstrMax(max_overhang, right_edges, name="LinkMaxEdge")
    model.setObjective(max_overhang, GRB.MAXIMIZE)
    model.update()

    return execute_and_report(model, X, L, T, N, method_label, time_limit_sec, random_seed, threads, mip_gap, eta=eta)
