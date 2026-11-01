#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_m1_order_prune.py
Model Variant M1 (M1_OrderPrune):
  - Upper-DAG topological reduction (1 <= j < i for i >= 2)
  - Unique table-root connectivity (L_1 = 1, T_{1,0} = 1; L_i >= 2, T_{i,0} = 0 for i >= 2)
  - Exact linearization of base reaction force F_{1,0} == N*m and base moment Torque_{1,0} == N*m*P_{1,0}
  - Two-support degree bound (1 <= sum_{1<=j<i} T_{i,j} <= 2) and layer-indexed force capacity u^F_i = (N - i + 1)*m
"""

import gurobipy as gp
from gurobipy import GRB
from maxOverhang_proposed import execute_and_report


def solve_m1_order_prune(N, time_limit_sec, random_seed, threads=1, mip_gap=5e-4, method_label="M1_OrderPrune", eta=0.0):
    """Build and solve Model M1 for an N-block overhang stack."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model(f"TowerOfLire_{method_label}", env=env)

    W = 0.5
    MASS = 1.0
    EPSILON = W * float(eta)
    MAX_X = float(N)
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
        L[i] = model.addVar(lb=2, ub=i, vtype=GRB.INTEGER, name=f"L_{i}")

    T = {}; F = {}; P = {}; Torque = {}; max_f_i = {}

    # Base block (i = 1) is the unique block on the table (j = 0): F_{1,0} = N * MASS is constant
    max_f_i[1] = float(N) * MASS
    T[1, 0] = model.addVar(lb=1.0, ub=1.0, vtype=GRB.BINARY, name="T_1_0")
    F[1, 0] = model.addVar(lb=max_f_i[1], ub=max_f_i[1], vtype=GRB.CONTINUOUS, name="F_1_0")
    P[1, 0] = model.addVar(lb=-2.0 * W + EPSILON, ub=-EPSILON, vtype=GRB.CONTINUOUS, name="P_1_0")
    Torque[1, 0] = model.addVar(lb=-max_f_i[1] * 2.0 * W, ub=-max_f_i[1] * EPSILON, vtype=GRB.CONTINUOUS, name="Torque_1_0")

    model.addConstr(P[1, 0] >= X[1] - W + EPSILON, name="BaseContact_LB")
    model.addConstr(P[1, 0] <= X[1] + W - EPSILON, name="BaseContact_UB")
    model.addConstr(Torque[1, 0] == max_f_i[1] * P[1, 0], name="BaseTorque_Linear")

    # Upper-DAG pairs: only 1 <= j < i exist for i >= 2
    for i in range(2, N + 1):
        max_f_i[i] = float(N - i + 1) * MASS
        for j in range(1, i):
            u_f = max_f_i[i]
            T[i, j] = model.addVar(vtype=GRB.BINARY, name=f"T_{i}_{j}")
            F[i, j] = model.addVar(lb=0.0, ub=u_f, vtype=GRB.CONTINUOUS, name=f"F_{i}_{j}")
            P[i, j] = model.addVar(lb=-MAX_X, ub=MAX_X, vtype=GRB.CONTINUOUS, name=f"P_{i}_{j}")
            Torque[i, j] = model.addVar(lb=-u_f * MAX_X, ub=u_f * MAX_X,
                                        vtype=GRB.CONTINUOUS, name=f"Torque_{i}_{j}")

    # 2. Geometric Contact and Upper-DAG Topological Constraints (for i >= 2)
    for i in range(2, N + 1):
        u_f = max_f_i[i]
        model.addConstr(gp.quicksum(T[i, j] for j in range(1, i)) >= 1, name=f"DAGSupMin_{i}")
        model.addConstr(gp.quicksum(T[i, j] for j in range(1, i)) <= 2, name=f"DAGSupMax_{i}")

        for j in range(1, i):
            model.addConstr(F[i, j] <= u_f * T[i, j], name=f"ForceCap_{i}_{j}")
            model.addConstr(P[i, j] >= -MAX_X * T[i, j], name=f"PCapLB_{i}_{j}")
            model.addConstr(P[i, j] <= MAX_X * T[i, j], name=f"PCapUB_{i}_{j}")
            model.addConstr(Torque[i, j] >= -u_f * MAX_X * T[i, j], name=f"TCapLB_{i}_{j}")
            model.addConstr(Torque[i, j] <= u_f * MAX_X * T[i, j], name=f"TCapUB_{i}_{j}")

            model.addConstr(P[i, j] >= X[i] - W + EPSILON - BIG_M * (1 - T[i, j]))
            model.addConstr(P[i, j] <= X[i] + W - EPSILON + BIG_M * (1 - T[i, j]))
            model.addConstr(P[i, j] >= X[j] - W + EPSILON - BIG_M * (1 - T[i, j]))
            model.addConstr(P[i, j] <= X[j] + W - EPSILON + BIG_M * (1 - T[i, j]))

            model.addConstr(L[i] - L[j] <= 1 + (i - 1) * (1 - T[i, j]), name=f"LayerUB_{i}_{j}")
            model.addConstr(L[i] - L[j] >= 1 - j * (1 - T[i, j]), name=f"LayerLB_{i}_{j}")
            model.addConstr(Torque[i, j] == F[i, j] * P[i, j], name=f"Bilinear_{i}_{j}")

    # 3. Same-Layer Non-Overlap and Layer Ordering Constraints
    for i in range(1, N):
        model.addConstr(L[i] <= L[i + 1], name=f"SymLayer_{i}")
        b = model.addVar(vtype=GRB.BINARY, name=f"LayerJump_{i}")
        model.addConstr(L[i + 1] - L[i] >= b)
        model.addConstr(L[i + 1] - L[i] <= i * b)
        model.addConstr(X[i] - X[i + 1] >= 2 * W - BIG_M * b, name=f"NonOverlap_{i}")

    # 4. Static Force and Moment Equilibrium Equations
    model.addConstr(gp.quicksum(X[i] for i in range(1, N + 1)) <= -float(N) * EPSILON, name="Global_CoM")

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
    right_edges = [model.addVar(lb=-MAX_X, ub=MAX_X, vtype=GRB.CONTINUOUS, name=f"RE_{i}") for i in range(1, N + 1)]
    for i in range(1, N + 1):
        model.addConstr(right_edges[i - 1] == X[i] + W)

    max_overhang = model.addVar(lb=0.5 - 2.0 * EPSILON, ub=MAX_X, vtype=GRB.CONTINUOUS, name="max_overhang")
    model.addGenConstrMax(max_overhang, right_edges, name="LinkMaxEdge")
    model.setObjective(max_overhang, GRB.MAXIMIZE)
    model.update()

    return execute_and_report(model, X, L, T, N, method_label, time_limit_sec, random_seed, threads, mip_gap, eta=eta)
