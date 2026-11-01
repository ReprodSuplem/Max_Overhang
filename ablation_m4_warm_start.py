#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_m4_warm_start.py
Model Variant M4 (M4_WarmStart):
  - Incorporates Model M3 polyhedral cuts, indicator-lifted 4D McCormick facets, recursive sub-stack cuts,
    and single-column counterweight tail collinearity cuts
  - Closed-form analytical-inversion multi-tower shielded spinal warm-start constructor
  - Regime-based certified minimum stack height domain pruning on L_N
  - Complete root-node MIP start initialization across all decision variables
"""

import math
import gurobipy as gp
from gurobipy import GRB
from ablation_m3_convex_hull import build_m3_model
from maxOverhang_proposed import execute_and_report


def _solve_loaded_spine_lemma35(N, k, eta=0.0):
    """Solve the loaded spinal optimality recurrence for spine height k, total weight N, and stability margin eta."""
    W = 0.5
    eps = W * float(eta)
    w_eff = W - eps
    L_eff = 2.0 * w_eff

    def eval_t1(t1):
        t = [0.0] * (k + 1)
        w = [0.0] * (k + 1)
        t[0] = 0.0
        t[1] = t1
        w[1] = t1 - 1.0
        for i in range(1, k):
            t_next = (t[i] ** 2) / (t[i - 1] + 0.5)
            if t_next < t[i] + 1.0:
                t_next = t[i] + 1.0
            t[i + 1] = t_next
            w[i + 1] = t[i + 1] - t[i] - 1.0
        return t, w

    lo, hi = 1.0, float(N)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        t, w = eval_t1(mid)
        if t[k] < float(N):
            lo = mid
        else:
            hi = mid
    t, w = eval_t1(0.5 * (lo + hi))
    d = [0.0] + [L_eff * (w[i] + 0.5) / t[i] for i in range(1, k + 1)]
    return sum(d[1:]), t, w, d


def _max_spinal_overhang_no_top_weight(N, k, eta=0.0):
    """
    Solve the loaded spinal recurrence where the topmost spine block at layer k carries zero external weight
    (w_1 = 0, t_1 = 1.0, d_1 = w_eff).
    """
    W = 0.5
    eps = W * float(eta)
    w_eff = W - eps
    L_eff = 2.0 * w_eff
    if k <= 1:
        return w_eff

    def eval_t2(t2):
        t = [0.0] * (k + 1)
        w = [0.0] * (k + 1)
        t[0] = 0.0
        t[1] = 1.0
        w[1] = 0.0
        t[2] = t2
        w[2] = t2 - 2.0
        for i in range(2, k):
            t_next = (t[i] ** 2) / (t[i - 1] + 0.5)
            if t_next < t[i] + 1.0:
                t_next = t[i] + 1.0
            t[i + 1] = t_next
            w[i + 1] = t[i + 1] - t[i] - 1.0
        return t, w

    lo, hi = 2.0, float(N)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        t, w = eval_t2(mid)
        if t[k] < float(N):
            lo = mid
        else:
            hi = mid
    t, w = eval_t2(0.5 * (lo + hi))
    d = [0.0] + [L_eff * (w[i] + 0.5) / t[i] for i in range(1, k + 1)]
    return sum(d[1:])


def compute_soda_certified_min_layers(N, target_overhang, eta=0.0):
    """
    Compute the certified minimum stack height lower bound L_N >= L_N^*:
      - Geometric minimum layer bound from X_1 <= ((N-1)/N)*w_eff - eps and step <= 2*w_eff per layer
      - Spinal optimality regime bound for N <= 19
    """
    W = 0.5
    eps = W * float(eta)
    w_eff = W - eps
    x1_ub = (float(N - 1) / float(N)) * w_eff - eps
    geom_min_l = max(2, int(math.ceil((target_overhang - W - x1_ub) / (2.0 * w_eff) + 1.0 - 1e-7)))
    if N > 19:
        return min(N, geom_min_l)

    for L_max in range(geom_min_l, N + 1):
        oh_no_top = _max_spinal_overhang_no_top_weight(N, L_max, eta=eta)
        oh_with_top, _, _, _ = _solve_loaded_spine_lemma35(N, L_max - 1, eta=eta)
        if max(oh_no_top, oh_with_top) >= target_overhang - 1e-5:
            return L_max
    return geom_min_l


def _check_closed_form_equilibrium(N, X_cand, L_cand, T_cand, F_cand, P_cand, W=0.5, MASS=1.0, eta=0.0, tol=1e-6):
    """Algebraic verifier for static force and moment balance, contact intervals, and same-layer non-overlap."""
    eps = W * float(eta)
    w_eff = W - eps
    if T_cand.get((1, 0), 0.0) < 0.5:
        return False
    p10 = P_cand.get((1, 0), 0.0)
    if p10 > -eps + 1e-7 or abs(p10 - X_cand[1]) > w_eff + 1e-6:
        return False

    by_layer = {}
    for i in range(1, N + 1):
        by_layer.setdefault(L_cand[i], []).append((i, X_cand[i]))
    for _, blks in by_layer.items():
        blks_sorted = sorted(blks, key=lambda item: -item[1])
        for idx in range(1, len(blks_sorted)):
            if blks_sorted[idx - 1][1] - blks_sorted[idx][1] < 2.0 * W - 1e-5:
                return False

    for i in range(1, N + 1):
        f_out = sum(F_cand.get((i, j), 0.0) for j in range(0, i) if T_cand.get((i, j), 0.0) > 0.5)
        f_in = sum(F_cand.get((k, i), 0.0) for k in range(i + 1, N + 1) if T_cand.get((k, i), 0.0) > 0.5) + MASS
        if abs(f_out - f_in) > tol:
            return False

        t_out = sum(F_cand.get((i, j), 0.0) * P_cand.get((i, j), 0.0)
                    for j in range(0, i) if T_cand.get((i, j), 0.0) > 0.5)
        t_in = sum(F_cand.get((k, i), 0.0) * P_cand.get((k, i), 0.0)
                   for k in range(i + 1, N + 1) if T_cand.get((k, i), 0.0) > 0.5) + MASS * X_cand[i]
        if abs(t_out - t_in) > tol:
            return False

        if i >= 2:
            sup_cnt = 0
            for j in range(1, i):
                if T_cand.get((i, j), 0.0) > 0.5:
                    sup_cnt += 1
                    if L_cand[i] != L_cand[j] + 1:
                        return False
                    pij = P_cand.get((i, j), 0.0)
                    if abs(pij - X_cand[i]) > w_eff + 1e-6 or abs(pij - X_cand[j]) > w_eff + 1e-6:
                        return False
            if sup_cnt < 1 or sup_cnt > 2:
                return False

    return True


def build_gap_aware_shielded_warm_start(N, W=0.5, MASS=1.0, eta=0.0):
    """
    Construct a balanced multi-tower shielded spinal DAG stack in closed form via analytical inversion
    for block count N and relative stability margin eta = epsilon / W >= 0.
    """
    eps = W * float(eta)
    w_eff = W - eps
    L_eff = 2.0 * w_eff
    best_sol = None

    # Candidate A: Direct Counterbalanced Spinal Stack
    for K in range(2, N + 1):
        M_bal = N - K
        if M_bal > K - 1:
            continue
        layer_map = {0: 0, 1: 1}
        spine_idx_at_layer = {1: 1}
        bal_idx_at_layer = {}
        curr_idx = 2
        for l in range(2, K + 1):
            spine_idx_at_layer[l] = curr_idx
            layer_map[curr_idx] = l
            curr_idx += 1
            if l <= M_bal + 1:
                bal_idx_at_layer[l] = curr_idx
                layer_map[curr_idx] = l
                curr_idx += 1
        T_cand = {}; F_cand = {}; P_cand = {}; X_cand = {0: 0.0}
        t_load = {}; w_left = {}
        for l in range(K, 0, -1):
            w_l = MASS if (l + 1) in bal_idx_at_layer else 0.0
            w_left[l] = w_l
            above_t = t_load[l + 1] if l < K else 0.0
            t_load[l] = above_t + MASS + w_l
        valid_cand = True
        for l in range(1, K + 1):
            s_id = spine_idx_at_layer[l]
            pivot_l = -eps if l == 1 else (X_cand[spine_idx_at_layer[l - 1]] + w_eff)
            above_t = t_load[l + 1] if l < K else 0.0
            w_l = w_left[l]
            X_cand[s_id] = pivot_l - (above_t - w_l) * w_eff / t_load[l]
            if abs(pivot_l - X_cand[s_id]) > w_eff + 1e-9:
                valid_cand = False
                break
            parent_id = 0 if l == 1 else spine_idx_at_layer[l - 1]
            T_cand[(s_id, parent_id)] = 1.0
            F_cand[(s_id, parent_id)] = t_load[l]
            P_cand[(s_id, parent_id)] = pivot_l
        if not valid_cand:
            continue
        for l in range(1, K):
            if (l + 1) in bal_idx_at_layer:
                b_id = bal_idx_at_layer[l + 1]
                s_parent = spine_idx_at_layer[l]
                s_sibling = spine_idx_at_layer[l + 1]
                p_contact = X_cand[s_parent] - w_eff
                X_cand[b_id] = min(p_contact, X_cand[s_sibling] - 2.0 * W)
                if abs(p_contact - X_cand[b_id]) > w_eff + 1e-9:
                    valid_cand = False
                    break
                if X_cand[s_sibling] - X_cand[b_id] < 2 * W - 1e-9:
                    valid_cand = False
                    break
                T_cand[(b_id, s_parent)] = 1.0
                F_cand[(b_id, s_parent)] = MASS
                P_cand[(b_id, s_parent)] = p_contact
        if valid_cand and _check_closed_form_equilibrium(N, X_cand, layer_map, T_cand, F_cand, P_cand, W, MASS, eta=eta):
            cand_oh = max(X_cand[i] + W for i in range(1, N + 1))
            if best_sol is None or cand_oh > best_sol["overhang"]:
                best_sol = {
                    "overhang": cand_oh, "X": X_cand, "L": layer_map,
                    "T": T_cand, "F": F_cand, "P": P_cand
                }

    # Candidate B: Shielded Spinal DAG Stack (primary tower on B'_1, secondary tower on B'_{idx_low})
    for k in range(2, (N + 1) // 2 + 1):
        if N < 2 * k:
            continue
        oh, t, w, d = _solve_loaded_spine_lemma35(N, k, eta=eta)
        if best_sol is not None and oh <= best_sol["overhang"]:
            continue
        if any(w[i] < 1e-5 for i in range(1, k + 1)):
            continue

        x = [0.0] * (k + 1)
        x[k] = -eps + d[k] - L_eff
        for i in range(k - 1, 0, -1):
            x[i] = x[i + 1] + d[i]

        V_tot = N - 2 * k
        if V_tot < 0:
            continue

        low_choices = [(2, 0)]
        if V_tot > 0:
            for idx_low in range(2, k):
                for v_low in range(1, V_tot + 1):
                    low_choices.append((idx_low, v_low))

        found_for_k = False
        for (idx_low, v_low) in low_choices:
            v1 = V_tot - v_low
            beta_steps = range(21) if v_low > 0 else [0]
            for s_alpha in [0, 2, 5, 10, 15, 20]:
                alpha = s_alpha / 20.0
                for s_beta in beta_steps:
                    beta = s_beta / 20.0 if v_low > 0 else alpha
                    y = [0.0] * (k + 1)
                    u = [0.0] * (k + 1)
                    z = [0.0] * (k + 1)
                    valid_shield = True
                    pv_low = 0.0

                    for i in range(k - 1, 1, -1):
                        hi_y = min(x[i] - 2.0 * W, x[i + 1])
                        lo_y = x[i + 1] - L_eff
                        if i < k - 1:
                            hi_y = min(hi_y, z[i + 1])
                            lo_y = max(lo_y, z[i + 1] - L_eff)
                        if v_low > 0 and i < idx_low:
                            layer_sh_i = k - i + 1
                            if (k - idx_low + 2) <= layer_sh_i <= (k - idx_low + 1 + v_low):
                                lo_y = max(lo_y, pv_low + 2.0 * W - w_eff)
                        if lo_y > hi_y + 1e-6:
                            valid_shield = False
                            break
                        frac_i = beta if (v_low > 0 and i == idx_low) else alpha
                        y[i] = hi_y - frac_i * (hi_y - lo_y)
                        u_raw = u[i + 1] + w[i + 1] - 1.0
                        rhs_m = z[i + 1] * u[i + 1] + x[i + 1] * w[i + 1] - (y[i] + w_eff)
                        if v_low > 0 and i == idx_low:
                            pv_low = y[i]
                            u[i] = u_raw - float(v_low)
                            if u[i] <= 1e-6:
                                valid_shield = False
                                break
                            z[i] = (rhs_m - float(v_low) * pv_low) / u[i]
                        else:
                            u[i] = u_raw
                            if u[i] <= 1e-6:
                                valid_shield = False
                                break
                            z[i] = rhs_m / u[i]
                        if not (y[i] - 1e-5 <= z[i] <= y[i] + L_eff + 1e-5):
                            valid_shield = False
                            break

                    if not valid_shield:
                        continue

                    u[1] = 1.0 - w[1]
                    if u[1] <= 1e-7:
                        continue

                    y1_min = x[2] - L_eff
                    y1_max = min(x[1] - 2.0 * W, x[2])
                    if k > 2:
                        y1_min = max(y1_min, z[2] - L_eff)
                        y1_max = min(y1_max, z[2])
                    if v_low > 0 and (k - idx_low + 2) <= k <= (k - idx_low + 1 + v_low):
                        y1_min = max(y1_min, pv_low + 2.0 * W - w_eff)

                    if y1_min > y1_max + 1e-6:
                        continue

                    chosen_y1 = 0.0; chosen_pv1 = 0.0; chosen_z1 = 0.0; chosen_y0 = 0.0
                    found_config = False
                    for s_y in range(51):
                        y1_c = y1_max - (y1_max - y1_min) * (s_y / 50.0)
                        rhs_m1 = (z[2] * u[2] if k > 2 else 0.0) + x[2] * w[2] - (y1_c + w_eff)

                        if v1 == 0:
                            z1_c = rhs_m1 / u[1]
                            y0_c = u[1] * z1_c + w[1] * x[1] - w_eff
                            ok_low_0 = True
                            if v_low > 0 and (k - idx_low + 2) <= (k + 1) <= (k - idx_low + 1 + v_low):
                                if pv_low + 2.0 * W - w_eff > y0_c + 1e-5:
                                    ok_low_0 = False
                            if (ok_low_0 and
                                    y1_c - 1e-5 <= z1_c <= y1_c + L_eff + 1e-5 and
                                    y0_c - 1e-5 <= z1_c <= y0_c + L_eff + 1e-5 and
                                    y0_c - 1e-5 <= x[1] <= y0_c + L_eff + 1e-5):
                                chosen_y1, chosen_pv1, chosen_z1, chosen_y0 = y1_c, y1_c, z1_c, y0_c
                                found_config = True
                                break
                        else:
                            z1_lo = y1_c
                            z1_hi = y1_c + L_eff
                            for s_z in range(51):
                                z1_c = z1_hi - (z1_hi - z1_lo) * (s_z / 50.0)
                                pv_c = (rhs_m1 - u[1] * z1_c) / float(v1)
                                y0_c = u[1] * z1_c + w[1] * x[1] - w_eff
                                ok_low_1 = True
                                if v_low > 0:
                                    max_l_low = k - idx_low + 1 + v_low
                                    if max_l_low >= k + 1 and pv_low + 2.0 * W > pv_c + 1e-5:
                                        ok_low_1 = False
                                if (ok_low_1 and
                                        y1_c - 1e-5 <= pv_c <= y1_c + L_eff + 1e-5 and
                                        y0_c - 1e-5 <= z1_c <= y0_c + L_eff + 1e-5 and
                                        y0_c - 1e-5 <= x[1] <= y0_c + L_eff + 1e-5 and
                                        pv_c + 2.0 * W - w_eff <= y0_c + 1e-5):
                                    chosen_y1, chosen_pv1, chosen_z1, chosen_y0 = y1_c, pv_c, z1_c, y0_c
                                    found_config = True
                                    break
                            if found_config:
                                break

                    if not found_config:
                        continue

                    y[1] = chosen_y1
                    z[1] = chosen_z1
                    y0 = chosen_y0
                    pv1 = chosen_pv1

                    blocks_by_layer = {}
                    for l in range(1, k + 1):
                        i_sp = k - l + 1
                        blocks_by_layer.setdefault(l, []).append(("spine", i_sp, x[i_sp] + w_eff))
                        if l >= 2:
                            blocks_by_layer[l].append(("shield", i_sp, y[i_sp] + w_eff))

                    blocks_by_layer.setdefault(k + 1, []).append(("bridge0", 0, y0 + w_eff))
                    for tau in range(1, v1 + 1):
                        blocks_by_layer.setdefault(k + tau, []).append(("tower1", tau, pv1))
                    if v_low > 0:
                        base_layer_low = k - idx_low + 1
                        for tau in range(1, v_low + 1):
                            blocks_by_layer.setdefault(base_layer_low + tau, []).append(("tower_low", tau, pv_low))

                    X_cand = {0: 0.0}
                    L_cand = {0: 0}
                    role_to_id = {}
                    curr_id = 1
                    overlap_violation = False

                    for l in sorted(blocks_by_layer.keys()):
                        row_blocks = sorted(blocks_by_layer[l], key=lambda item: -item[2])
                        for idx_in_row, (role, sub_idx, cx) in enumerate(row_blocks):
                            if idx_in_row > 0 and row_blocks[idx_in_row - 1][2] - cx < 2.0 * W - 1e-5:
                                overlap_violation = True
                                break
                            X_cand[curr_id] = cx
                            L_cand[curr_id] = l
                            role_to_id[(role, sub_idx)] = curr_id
                            curr_id += 1
                        if overlap_violation:
                            break

                    if overlap_violation or curr_id != N + 1:
                        continue

                    T_cand = {}; F_cand = {}; P_cand = {}
                    for i_sp in range(1, k + 1):
                        u_id = role_to_id[("spine", i_sp)]
                        v_id = 0 if i_sp == k else role_to_id[("spine", i_sp + 1)]
                        piv = -eps if i_sp == k else (x[i_sp + 1] + L_eff)
                        T_cand[(u_id, v_id)] = 1.0
                        F_cand[(u_id, v_id)] = t[i_sp] * MASS
                        P_cand[(u_id, v_id)] = piv

                    for i_sh in range(1, k):
                        sh_id = role_to_id[("shield", i_sh)]
                        sp_below = role_to_id[("spine", i_sh + 1)]
                        T_cand[(sh_id, sp_below)] = 1.0
                        F_cand[(sh_id, sp_below)] = w[i_sh + 1] * MASS
                        P_cand[(sh_id, sp_below)] = x[i_sh + 1]

                        if i_sh < k - 1:
                            sh_below = role_to_id[("shield", i_sh + 1)]
                            T_cand[(sh_id, sh_below)] = 1.0
                            F_cand[(sh_id, sh_below)] = u[i_sh + 1] * MASS
                            P_cand[(sh_id, sh_below)] = z[i_sh + 1]

                    b0_id = role_to_id[("bridge0", 0)]
                    sp1_id = role_to_id[("spine", 1)]
                    sh1_id = role_to_id[("shield", 1)]
                    T_cand[(b0_id, sp1_id)] = 1.0
                    F_cand[(b0_id, sp1_id)] = w[1] * MASS
                    P_cand[(b0_id, sp1_id)] = x[1]

                    T_cand[(b0_id, sh1_id)] = 1.0
                    F_cand[(b0_id, sh1_id)] = u[1] * MASS
                    P_cand[(b0_id, sh1_id)] = z[1]

                    for tau in range(1, v1 + 1):
                        tw_id = role_to_id[("tower1", tau)]
                        below_id = sh1_id if tau == 1 else role_to_id[("tower1", tau - 1)]
                        rem_load = float(v1 - tau + 1) * MASS
                        T_cand[(tw_id, below_id)] = 1.0
                        F_cand[(tw_id, below_id)] = rem_load
                        P_cand[(tw_id, below_id)] = pv1

                    if v_low > 0:
                        sh_low_id = role_to_id[("shield", idx_low)]
                        for tau in range(1, v_low + 1):
                            tw2_id = role_to_id[("tower_low", tau)]
                            below2_id = sh_low_id if tau == 1 else role_to_id[("tower_low", tau - 1)]
                            rem2_load = float(v_low - tau + 1) * MASS
                            T_cand[(tw2_id, below2_id)] = 1.0
                            F_cand[(tw2_id, below2_id)] = rem2_load
                            P_cand[(tw2_id, below2_id)] = pv_low

                    if _check_closed_form_equilibrium(N, X_cand, L_cand, T_cand, F_cand, P_cand, W, MASS, eta=eta):
                        cand_oh = max(X_cand[i] + W for i in range(1, N + 1))
                        if best_sol is None or cand_oh > best_sol["overhang"]:
                            best_sol = {
                                "overhang": cand_oh,
                                "X": X_cand,
                                "L": L_cand,
                                "T": T_cand,
                                "F": F_cand,
                                "P": P_cand
                            }
                        found_for_k = True
                        break
                if found_for_k:
                    break
            if found_for_k:
                break

    # Candidate C: Top-Bridge-Supported Counterweight Tower (under positive stability margin eta > 0)
    if eta > 0.0:
        for k in range(2, (N + 1) // 2 + 1):
            if N < 2 * k:
                continue
            oh, t, w, d = _solve_loaded_spine_lemma35(N, k, eta=eta)
            if best_sol is not None and oh <= best_sol["overhang"] + 1e-9:
                continue
            if any(w[i] < 1e-5 for i in range(1, k + 1)):
                continue

            x = [0.0] * (k + 1)
            x[k] = -eps + d[k] - L_eff
            for i in range(k - 1, 0, -1):
                x[i] = x[i + 1] + d[i]

            V_tot = N - 2 * k
            for v1 in range(1, V_tot + 1):
                v_low = V_tot - v1
                if v_low > 0 and k <= 2:
                    continue
                u1 = float(1 + v1) - w[1]
                if u1 <= 1e-6:
                    continue
                low_idx_list = range(2, k) if v_low > 0 else [2]
                found_c = False
                for idx_low in low_idx_list:
                    for s_alpha in range(21):
                        alpha = s_alpha / 20.0
                        y = [0.0] * (k + 1)
                        u = [0.0] * (k + 1)
                        z = [0.0] * (k + 1)
                        valid_sh = True
                        pv_low = 0.0
                        for i in range(k - 1, 1, -1):
                            hi_y = min(x[i] - 2.0 * W, x[i + 1])
                            lo_y = x[i + 1] - L_eff
                            if i < k - 1:
                                hi_y = min(hi_y, z[i + 1])
                                lo_y = max(lo_y, z[i + 1] - L_eff)
                            if v_low > 0 and i < idx_low:
                                layer_sh_i = k - i + 1
                                if (k - idx_low + 2) <= layer_sh_i <= (k - idx_low + 1 + v_low):
                                    lo_y = max(lo_y, pv_low + 2.0 * W - w_eff)
                            if lo_y > hi_y + 1e-6:
                                valid_sh = False
                                break
                            y[i] = hi_y - alpha * (hi_y - lo_y)
                            u_raw = u[i + 1] + w[i + 1] - 1.0
                            rhs_m = z[i + 1] * u[i + 1] + x[i + 1] * w[i + 1] - (y[i] + w_eff)
                            if v_low > 0 and i == idx_low:
                                pv_low = y[i]
                                u[i] = u_raw - float(v_low)
                                if u[i] <= 1e-6:
                                    valid_sh = False
                                    break
                                z[i] = (rhs_m - float(v_low) * pv_low) / u[i]
                            else:
                                u[i] = u_raw
                                if u[i] <= 1e-6:
                                    valid_sh = False
                                    break
                                z[i] = rhs_m / u[i]
                            if not (y[i] - 1e-5 <= z[i] <= y[i] + L_eff + 1e-5):
                                valid_sh = False
                                break
                        if not valid_sh:
                            continue

                        y1_min = x[2] - L_eff
                        y1_max = min(x[1] - 2.0 * W, x[2])
                        if k > 2:
                            y1_min = max(y1_min, z[2] - L_eff)
                            y1_max = min(y1_max, z[2])
                        if v_low > 0 and (k - idx_low + 2) <= k <= (k - idx_low + 1 + v_low):
                            y1_min = max(y1_min, pv_low + 2.0 * W - w_eff)
                        if y1_min > y1_max + 1e-6:
                            continue

                        for s_y in range(51):
                            y1_c = y1_max - (y1_max - y1_min) * (s_y / 50.0)
                            rhs_m1 = (z[2] * u[2] if k > 2 else 0.0) + x[2] * w[2] - (y1_c + w_eff)
                            z1_c = rhs_m1 / u1
                            if not (y1_c - 1e-5 <= z1_c <= y1_c + L_eff + 1e-5):
                                continue
                            tot_m0 = u1 * z1_c + w[1] * x[1]
                            y0_lo = max(z1_c - L_eff, x[1] - L_eff)
                            y0_hi = min(z1_c, x[1])
                            if v_low > 0 and (k - idx_low + 2) <= (k + 1) <= (k - idx_low + 1 + v_low):
                                y0_lo = max(y0_lo, pv_low + 2.0 * W - w_eff)
                            if y0_lo > y0_hi + 1e-6:
                                continue
                            found_y0 = False
                            p_top_c = 0.0
                            y0_c = 0.0
                            for s_0 in range(21):
                                y0_cand = y0_hi - (y0_hi - y0_lo) * (s_0 / 20.0)
                                p_top_cand = (tot_m0 - (y0_cand + w_eff)) / float(v1)
                                if y0_cand - 1e-5 <= p_top_cand <= y0_cand + L_eff + 1e-5:
                                    y0_c = y0_cand
                                    p_top_c = p_top_cand
                                    found_y0 = True
                                    break
                            if not found_y0:
                                continue

                            blocks_by_layer = {}
                            for l in range(1, k + 1):
                                i_sp = k - l + 1
                                blocks_by_layer.setdefault(l, []).append(("spine", i_sp, x[i_sp] + w_eff))
                                if l >= 2:
                                    blocks_by_layer[l].append(("shield", i_sp, (y1_c if i_sp == 1 else y[i_sp]) + w_eff))
                            blocks_by_layer.setdefault(k + 1, []).append(("bridge0", 0, y0_c + w_eff))
                            for tau in range(1, v1 + 1):
                                blocks_by_layer.setdefault(k + 1 + tau, []).append(("tower_top", tau, p_top_c))
                            if v_low > 0:
                                base_layer_low = k - idx_low + 1
                                for tau in range(1, v_low + 1):
                                    blocks_by_layer.setdefault(base_layer_low + tau, []).append(("tower_low", tau, pv_low))

                            X_cand = {0: 0.0}; L_cand = {0: 0}; role_to_id = {}; curr_id = 1; ov_viol = False
                            for l in sorted(blocks_by_layer.keys()):
                                row_blocks = sorted(blocks_by_layer[l], key=lambda item: -item[2])
                                for idx_in_row, (role, sub_idx, cx) in enumerate(row_blocks):
                                    if idx_in_row > 0 and row_blocks[idx_in_row - 1][2] - cx < 2.0 * W - 1e-5:
                                        ov_viol = True
                                        break
                                    X_cand[curr_id] = cx
                                    L_cand[curr_id] = l
                                    role_to_id[(role, sub_idx)] = curr_id
                                    curr_id += 1
                                if ov_viol:
                                    break
                            if ov_viol or curr_id != N + 1:
                                continue

                            T_cand = {}; F_cand = {}; P_cand = {}
                            for i_sp in range(1, k + 1):
                                u_id = role_to_id[("spine", i_sp)]
                                v_id = 0 if i_sp == k else role_to_id[("spine", i_sp + 1)]
                                piv = -eps if i_sp == k else (x[i_sp + 1] + L_eff)
                                T_cand[(u_id, v_id)] = 1.0
                                F_cand[(u_id, v_id)] = t[i_sp] * MASS
                                P_cand[(u_id, v_id)] = piv

                            for i_sh in range(1, k):
                                sh_id = role_to_id[("shield", i_sh)]
                                sp_below = role_to_id[("spine", i_sh + 1)]
                                T_cand[(sh_id, sp_below)] = 1.0
                                F_cand[(sh_id, sp_below)] = w[i_sh + 1] * MASS
                                P_cand[(sh_id, sp_below)] = x[i_sh + 1]
                                if i_sh < k - 1:
                                    sh_below = role_to_id[("shield", i_sh + 1)]
                                    T_cand[(sh_id, sh_below)] = 1.0
                                    F_cand[(sh_id, sh_below)] = u[i_sh + 1] * MASS
                                    P_cand[(sh_id, sh_below)] = z[i_sh + 1]

                            b0_id = role_to_id[("bridge0", 0)]
                            sp1_id = role_to_id[("spine", 1)]
                            sh1_id = role_to_id[("shield", 1)]
                            T_cand[(b0_id, sp1_id)] = 1.0
                            F_cand[(b0_id, sp1_id)] = w[1] * MASS
                            P_cand[(b0_id, sp1_id)] = x[1]
                            T_cand[(b0_id, sh1_id)] = 1.0
                            F_cand[(b0_id, sh1_id)] = u1 * MASS
                            P_cand[(b0_id, sh1_id)] = z1_c

                            for tau in range(1, v1 + 1):
                                tw_id = role_to_id[("tower_top", tau)]
                                below_id = b0_id if tau == 1 else role_to_id[("tower_top", tau - 1)]
                                rem_load = float(v1 - tau + 1) * MASS
                                T_cand[(tw_id, below_id)] = 1.0
                                F_cand[(tw_id, below_id)] = rem_load
                                P_cand[(tw_id, below_id)] = p_top_c

                            if v_low > 0:
                                sh_low_id = role_to_id[("shield", idx_low)]
                                for tau in range(1, v_low + 1):
                                    tw2_id = role_to_id[("tower_low", tau)]
                                    below2_id = sh_low_id if tau == 1 else role_to_id[("tower_low", tau - 1)]
                                    rem2_load = float(v_low - tau + 1) * MASS
                                    T_cand[(tw2_id, below2_id)] = 1.0
                                    F_cand[(tw2_id, below2_id)] = rem2_load
                                    P_cand[(tw2_id, below2_id)] = pv_low

                            if _check_closed_form_equilibrium(N, X_cand, L_cand, T_cand, F_cand, P_cand, W, MASS, eta=eta):
                                cand_oh = max(X_cand[i] + W for i in range(1, N + 1))
                                if best_sol is None or cand_oh > best_sol["overhang"]:
                                    best_sol = {
                                        "overhang": cand_oh, "X": X_cand, "L": L_cand,
                                        "T": T_cand, "F": F_cand, "P": P_cand
                                    }
                                found_c = True
                                break
                        if found_c:
                            break
                    if found_c:
                        break

    return best_sol


def solve_m4_warm_start(N, time_limit_sec, random_seed, threads=1, mip_gap=5e-4, eta=0.0):
    """Build and solve Model M4 for an N-block overhang stack."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()

    W = 0.5
    model, X, L, T, F, P, Torque, b_jump, right_edges, max_overhang = build_m3_model(
        N, env, "TowerOfLire_M4_WarmStart", enable_lifted_hull=True, eta=eta
    )
    model.update()

    ws = build_gap_aware_shielded_warm_start(N, W=W, MASS=1.0, eta=eta)
    if ws is not None:
        min_layer_lb = compute_soda_certified_min_layers(N, ws["overhang"], eta=eta)
        print(f"INFO: Injecting Analytical Warm-Start (Eta = {eta:.4f}, Overhang = {ws['overhang']:.5f}) | "
              f"Certified Min Layer L_{N} >= {min_layer_lb}")

        # 1. Certified Minimum Layer-Height Pruning on L[N].LB for N >= 8
        if N >= 8 and min_layer_lb <= N:
            L[N].LB = max(2, min_layer_lb)

        # 2. Complete Root-Node MIP Start Assignment
        X[0].Start = 0.0
        L[0].Start = 0
        max_overhang.Start = ws["overhang"]

        for i in range(1, N + 1):
            X[i].Start = ws["X"][i]
            L[i].Start = ws["L"][i]
            right_edges[i - 1].Start = ws["X"][i] + W
            if i < N:
                bj_val = float(1 if ws["L"][i + 1] > ws["L"][i] else 0)
                b_jump[i].Start = bj_val

        f10_val = ws["F"].get((1, 0), float(N))
        p10_val = ws["P"].get((1, 0), -W * float(eta))
        T[1, 0].Start = 1.0
        F[1, 0].Start = f10_val
        P[1, 0].Start = p10_val
        Torque[1, 0].Start = f10_val * p10_val

        for i in range(2, N + 1):
            for j in range(1, i):
                t_val = ws["T"].get((i, j), 0.0)
                f_val = ws["F"].get((i, j), 0.0)
                p_val = ws["P"].get((i, j), 0.0)
                T[i, j].Start = t_val
                F[i, j].Start = f_val if t_val > 0.5 else 0.0
                P[i, j].Start = p_val if t_val > 0.5 else 0.0
                Torque[i, j].Start = (f_val * p_val) if t_val > 0.5 else 0.0

        model.update()

    return execute_and_report(model, X, L, T, N, "M4_WarmStart", time_limit_sec, random_seed, threads, mip_gap, eta=eta)
