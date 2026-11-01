#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_m3_convex_hull.py
Model Variant M3 (M3_ConvexHull):
  - Incorporates Model M2 domain tightening and disjunctive contact-interval hull
  - Macro-block free-body overturning envelope:
      (N - 1)*(X_1 - w_eff) <= sum_{i=2}^N X_i <= (N - 1)*(X_1 + w_eff)
    and base coordinate upper bound X_1 <= ((N - 1) / N) * w_eff - EPSILON
  - First-level RLT base moment cone bounds
  - Sparsity-preserving layer-jump inequality (b_jump[i] <= L_{i+1} - L_i <= min(i, 2)*b_jump[i], b_jump[1] == 1)
    and adjacent-support locking cut (T_{i+1, i} <= b_jump[i])
  - Indicator-lifted four-dimensional McCormick tetrahedral facets, recursive sub-stack cuts,
    and single-column counterweight tail collinearity cuts (enabled in Model M4 via enable_lifted_hull)
"""

import math
import gurobipy as gp
from gurobipy import GRB
from ablation_m2_local_moment import _compute_recursive_substack_ub
from maxOverhang_proposed import execute_and_report


def build_m3_model(N, env, model_name="TowerOfLire_M3_ConvexHull", enable_lifted_hull=False, eta=0.0):
    """Construct and return the polyhedral-strengthened MIBLP model and its variable dictionaries."""
    model = gp.Model(model_name, env=env)

    W = 0.5
    MASS = 1.0
    EPSILON = W * float(eta)
    w_eff = W - EPSILON

    sub_stack_ub = _compute_recursive_substack_ub(N - 1, eta=eta)
    base_ub = min(float(N) * w_eff + W, 4.5 * math.pow(N, 1.0 / 3.0) + 1.0)
    if enable_lifted_hull or N >= 14:
        theo_overhang_ub = min(base_ub, (float(N - 1) / float(N)) * w_eff - EPSILON + W + sub_stack_ub)
        MAX_X = max(2.0, min(float(N), theo_overhang_ub + 0.5))
    else:
        theo_overhang_ub = base_ub
        MAX_X = max(2.0, min(float(N), theo_overhang_ub))

    # Base coordinate domain from macro-block equilibrium: X_1 <= ((N-1)/N)*w_eff - EPSILON
    x_lb = {0: 0.0, 1: -W + EPSILON}
    x_ub = {0: 0.0, 1: (float(N - 1) / float(N)) * w_eff - EPSILON}
    for i in range(2, N + 1):
        if enable_lifted_hull:
            x_lb[i] = max(-MAX_X, x_lb[1] - (i - 1) * (2.0 * W))
            x_ub[i] = min(MAX_X - W, x_ub[1] + (i - 1) * (2.0 * w_eff), x_ub[1] + sub_stack_ub, theo_overhang_ub - W)
        else:
            x_lb[i] = -MAX_X
            x_ub[i] = min(MAX_X - W, W + (i - 1) * (2.0 * w_eff), theo_overhang_ub - W)

    # 1. Decision Variables (L_1 = 1, L_i in [2, i] for i >= 2)
    X = {}
    X[0] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name="X_0")
    for i in range(1, N + 1):
        X[i] = model.addVar(lb=x_lb[i], ub=x_ub[i], vtype=GRB.CONTINUOUS, name=f"X_{i}")

    L = {}
    L[0] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name="L_0")
    L[1] = model.addVar(lb=1, ub=1, vtype=GRB.INTEGER, name="L_1")
    for i in range(2, N + 1):
        ub_l = (2 if i == 2 else i) if enable_lifted_hull else i
        L[i] = model.addVar(lb=2, ub=ub_l, vtype=GRB.INTEGER, name=f"L_{i}")

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

            t_low_bound = 1.0 if (enable_lifted_hull and i == 2 and j == 1) else 0.0
            T[i, j] = model.addVar(lb=t_low_bound, ub=1.0, vtype=GRB.BINARY, name=f"T_{i}_{j}")
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
        if enable_lifted_hull or N >= 14:
            model.addConstr(X[i] - X[1] <= sub_stack_ub, name=f"RecSubStack_{i}")

        for j in range(1, i):
            l_p = p_lb[i, j]
            u_p = p_ub[i, j]

            model.addConstr(F[i, j] <= u_f * T[i, j], name=f"ForceCap_{i}_{j}")
            model.addConstr(P[i, j] >= l_p * T[i, j], name=f"PCapLB_{i}_{j}")
            model.addConstr(P[i, j] <= u_p * T[i, j], name=f"PCapUB_{i}_{j}")

            if enable_lifted_hull:
                model.addConstr(Torque[i, j] >= l_p * F[i, j], name=f"McLB1_{i}_{j}")
                model.addConstr(Torque[i, j] <= u_p * F[i, j], name=f"McUB1_{i}_{j}")
                model.addConstr(Torque[i, j] >= u_p * F[i, j] + u_f * P[i, j] - u_f * u_p * T[i, j],
                                name=f"McLB2_{i}_{j}")
                model.addConstr(Torque[i, j] <= l_p * F[i, j] + u_f * P[i, j] - u_f * l_p * T[i, j],
                                name=f"McUB2_{i}_{j}")
            else:
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

    # 3. Sparsity-Preserving Layer-Jump Inequality and Adjacent-Support Locking Cut
    b_jump = {}
    for i in range(1, N):
        lb_b = 1.0 if i == 1 else 0.0
        b_jump[i] = model.addVar(lb=lb_b, ub=1.0, vtype=GRB.BINARY, name=f"LayerJump_{i}")
        model.addConstr(L[i] <= L[i + 1], name=f"SymLayer_{i}")
        model.addConstr(L[i + 1] - L[i] >= b_jump[i], name=f"LayerJumpLB_{i}")
        model.addConstr(L[i + 1] - L[i] <= min(i, 2) * b_jump[i], name=f"LayerJumpUB_{i}")
        span_m = (x_ub[i + 1] - x_lb[i]) + 2.0 * W
        model.addConstr(X[i] - X[i + 1] >= 2 * W - span_m * b_jump[i], name=f"NonOverlap_{i}")
        if i >= 2:
            model.addConstr(T[i + 1, i] <= b_jump[i], name=f"AdjSupLock_{i}")

    # 3b. Single-Column Counterweight Tail Collinearity & Contact Collapse Cuts (for Model M4)
    if enable_lifted_hull:
        for i in range(3, N + 1):
            tail_no_jump = gp.quicksum(1 - b_jump[k] for k in range(i - 2, N))
            span_p = max(abs(p_ub[i, i - 1] - x_lb[i]), abs(p_lb[i, i - 1] - x_ub[i]), abs(x_ub[i]), abs(x_lb[i])) + 2.0 * W
            model.addConstr(P[i, i - 1] - X[i] <= span_p * tail_no_jump, name=f"TailContactAlignUB_{i}")
            model.addConstr(P[i, i - 1] - X[i] >= -span_p * tail_no_jump, name=f"TailContactAlignLB_{i}")
            if i >= 4:
                is_two_sup_below = gp.quicksum(T[i - 1, j] for j in range(1, i - 1)) - 1
                span_align = max(abs(x_ub[i] - x_lb[i - 1]), abs(x_ub[i - 1] - x_lb[i]))
                model.addConstr(X[i] - X[i - 1] <= span_align * (is_two_sup_below + tail_no_jump),
                                name=f"TailCollinearUB_{i}")
                model.addConstr(X[i] - X[i - 1] >= -span_align * (is_two_sup_below + tail_no_jump),
                                name=f"TailCollinearLB_{i}")

    # 4. Static Equilibrium Network with Macro-Block Overturning and Base RLT Envelopes
    model.addConstr(gp.quicksum(X[i] for i in range(1, N + 1)) <= -float(N) * EPSILON, name="Global_CoM")
    model.addConstr(gp.quicksum(MASS * X[i] for i in range(1, N + 1)) == Torque[1, 0],
                    name="Global_Table_Torque_Identity")
    model.addConstr(Torque[1, 0] >= float(N) * MASS * (X[1] - w_eff), name="BaseRLT_ExactLB")
    model.addConstr(Torque[1, 0] <= float(N) * MASS * (X[1] + w_eff), name="BaseRLT_ExactUB")

    sum_upper_x = gp.quicksum(X[i] for i in range(2, N + 1))
    model.addConstr(sum_upper_x >= float(N - 1) * (X[1] - w_eff), name="MacroBlock2_LB")
    model.addConstr(sum_upper_x <= float(N - 1) * (X[1] + w_eff), name="MacroBlock2_UB")

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

    return model, X, L, T, F, P, Torque, b_jump, right_edges, max_overhang


def solve_m3_convex_hull(N, time_limit_sec, random_seed, threads=1, mip_gap=5e-4, method_label="M3_ConvexHull", eta=0.0):
    """Build and solve Model M3 for an N-block overhang stack."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model, X, L, T, _, _, _, _, _, _ = build_m3_model(
        N, env, f"TowerOfLire_{method_label}", enable_lifted_hull=False, eta=eta
    )
    return execute_and_report(model, X, L, T, N, method_label, time_limit_sec, random_seed, threads, mip_gap, eta=eta)
