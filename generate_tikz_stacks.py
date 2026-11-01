#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_tikz_stacks.py
LaTeX TikZ Visual Atlas Generator for the Maximum Overhang Problem (.out logs -> appendix.tex).

Strictly parses actual Gurobi solver `.out` log files from:
  - `--ablation-dir` (default: results_N6_35) for eta = 0.00
  - `--pareto-dir`   (default: results_pareto) for eta in {0.01, 0.02, 0.05, 0.10}
Only panels with existing `.out` files on disk are rendered. No cached data or analytical
fallbacks are used.
"""

import os
import re
import sys
import argparse


def parse_stack_from_out(filepath):
    """Parse STACK_CONFIGURATION and ObjVal from a Gurobi .out log file on disk."""
    if not os.path.exists(filepath):
        return None, None

    obj_val = None
    blocks = []
    in_stack = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("ObjVal") and ":" in s:
                try:
                    obj_val = float(s.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif s.startswith("STACK_CONFIGURATION:"):
                in_stack = True
                continue
            elif in_stack:
                m = re.match(
                    r"Layer\s+(\d+)\s*\|\s*Block\s+(\d+)\s*\|\s*Center X:\s*([0-9.eE+-]+)\s*\|\s*Supported by:\s*\[([^\]]*)\]",
                    s
                )
                if m:
                    layer = int(m.group(1))
                    b_id = int(m.group(2))
                    cx = float(m.group(3))
                    sup_str = m.group(4).strip()
                    sups = [int(x.strip()) for x in sup_str.split(",") if x.strip()] if sup_str else []
                    blocks.append((layer, cx, b_id, sups))
                elif s.startswith("---") or s.startswith("INFO:"):
                    break

    if not blocks:
        return None, None
    if obj_val is None:
        obj_val = max(cx + 0.5 for _, cx, _, _ in blocks)
    return blocks, obj_val


def classify_roles(blocks):
    """
    Classify each block into one of four structural roles based on the support DAG:
      - 'spine'  : primary load-bearing support spine from principal overhanging block down to B_1
      - 'bridge' : top bridge block resting on both the topmost spine block and top shield block
      - 'shield' : shield blocks adjacent to the spine that transfer counterweight loads
      - 'tower'  : vertical counterweight tower blocks resting above the shield/bridge structure
    """
    by_id = {b_id: {"layer": lyr, "x": cx, "supports": sups} for lyr, cx, b_id, sups in blocks}
    principal_id = max(by_id.keys(), key=lambda i: (round(by_id[i]["x"], 6), by_id[i]["layer"]))

    spine_ids = set([principal_id])
    curr = principal_id
    while by_id[curr]["layer"] > 1:
        sups = [j for j in by_id[curr]["supports"] if j in by_id]
        if not sups:
            break
        nxt = max(sups, key=lambda j: by_id[j]["x"])
        spine_ids.add(nxt)
        curr = nxt
    if 1 in by_id:
        spine_ids.add(1)

    bridge_ids = set()
    for b_id, info in by_id.items():
        if b_id not in spine_ids and principal_id in info["supports"] and len(info["supports"]) >= 2:
            bridge_ids.add(b_id)

    shield_ids = set()
    for b_id, info in by_id.items():
        if b_id not in spine_ids and b_id not in bridge_ids:
            if any(s in spine_ids for s in info["supports"]):
                shield_ids.add(b_id)

    roles = {}
    for b_id in by_id:
        if b_id in spine_ids:
            roles[b_id] = "spine"
        elif b_id in bridge_ids:
            roles[b_id] = "bridge"
        elif b_id in shield_ids:
            roles[b_id] = "shield"
        else:
            roles[b_id] = "tower"
    return roles, principal_id


def render_single_panel_tikz(blocks, obj_val, panel_title, W=0.5, target_h=2.65, bh_max=0.22):
    """Generate TikZ commands for a single stack configuration panel."""
    roles, _ = classify_roles(blocks)
    max_layer = max(lyr for lyr, _, _, _ in blocks)
    min_x = min(cx - W for _, cx, _, _ in blocks)
    max_x = max(cx + W for _, cx, _, _ in blocks)

    span_x = max(2.8, (max_x - min_x) + 0.6)
    x_scale = min(1.80, 6.0 / span_x)
    bh = min(bh_max, target_h / max(5.0, float(max_layer + 1)))

    lines = []
    lines.append(r"\begin{tikzpicture}[x=" + f"{x_scale:.3f}cm, y=1.0cm, font=\\scriptsize]")
    lines.append(f"  \\path (0.000, {(max_layer + 0.92)*bh:.3f});")

    tbl_left = min(-1.35, min_x - 0.25)
    tbl_right = 0.0
    lines.append(f"  \\fill[gray!25] ({tbl_left:.3f}, {-1.15*bh:.3f}) rectangle ({tbl_right:.3f}, 0.000);")
    lines.append(f"  \\draw[thick, black] ({tbl_left:.3f}, 0.000) -- ({tbl_right:.3f}, 0.000) -- ({tbl_right:.3f}, {-1.15*bh:.3f});")
    lines.append(f"  \\draw[dashed, black!60] (0.000, {-1.15*bh:.3f}) -- (0.000, {(max_layer + 0.75)*bh:.3f});")

    sorted_blocks = sorted(blocks, key=lambda item: (item[0], item[1]))
    for lyr, cx, b_id, _ in sorted_blocks:
        r_type = roles.get(b_id, "tower")
        if r_type == "spine":
            style = "fill=blue!22, draw=blue!75!black, thick"
        elif r_type == "shield":
            style = "fill=orange!28, draw=orange!80!black, semithick"
        elif r_type == "bridge":
            style = "fill=teal!25, draw=teal!80!black, semithick"
        else:
            style = "fill=purple!18, draw=purple!70!black, semithick"

        x1 = cx - W
        x2 = cx + W
        y1 = (lyr - 1) * bh
        y2 = lyr * bh
        lines.append(f"  \\filldraw[{style}] ({x1:.4f}, {y1:.4f}) rectangle ({x2:.4f}, {y2:.4f});")

        if len(blocks) <= 13:
            ym = 0.5 * (y1 + y2)
            lines.append(f"  \\node[font=\\tiny, inner sep=0.2pt] at ({cx:.4f}, {ym:.4f}) {{{b_id}}};")

    lines.append(f"  \\draw[densely dotted, red!80!black, thick] ({obj_val:.4f}, 0.000) -- ({obj_val:.4f}, {(max_layer + 0.55)*bh:.3f});")
    lines.append(f"  \\node[anchor=south, font=\\scriptsize, text=red!80!black, inner sep=1pt] at ({obj_val:.4f}, {(max_layer + 0.57)*bh:.3f}) {{$z={obj_val:.4f}$}};")
    lines.append(f"  \\node[anchor=north, font=\\scriptsize\\bfseries] at ({0.5*(tbl_left + max_x):.3f}, {-1.30*bh:.3f}) {{{panel_title}}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def build_grid_figure_from_logs(panel_specs, caption_text, label_name, note_text, stats_counter, target_h=2.65, bh_max=0.22, row_sep="6pt"):
    """
    Build a 2-column INFORMS figure environment strictly from existing .out log files on disk.
    Panels whose .out log files do not exist are skipped with a warning.
    """
    panel_tikz_list = []
    for _, n_val, eta_val, fpath in panel_specs:
        stats_counter["requested"] += 1
        blocks, obj_val = parse_stack_from_out(fpath)
        if blocks is None:
            stats_counter["missing"].append(fpath)
            continue
        stats_counter["rendered"] += 1
        sub_lbl = chr(ord("a") + len(panel_tikz_list))
        max_l = max(b[0] for b in blocks)
        title_str = rf"({sub_lbl}) $N={n_val}, \eta={eta_val:.2f}, L_{{{n_val}}}={max_l}$"
        panel_tikz_list.append(render_single_panel_tikz(blocks, obj_val, title_str, target_h=target_h, bh_max=bh_max))

    if not panel_tikz_list:
        return None

    rows_tex = []
    num_rows = (len(panel_tikz_list) + 1) // 2
    for r_idx in range(num_rows):
        left_p = panel_tikz_list[2 * r_idx]
        right_p = panel_tikz_list[2 * r_idx + 1] if (2 * r_idx + 1) < len(panel_tikz_list) else ""
        row_str = left_p + " &\n" + right_p if right_p else left_p + " &"
        if r_idx < num_rows - 1:
            rows_tex.append(row_str + f" \\\\[{row_sep}]")
        else:
            rows_tex.append(row_str)

    return "\n".join([
        r"\begin{figure}[ht!]",
        r"\FIGURE",
        r"{\begin{tabular}{@{}c@{\quad}c@{}}\\[-6pt]",
        "\n".join(rows_tex),
        r"\end{tabular}}",
        rf"{{{caption_text}\label{{{label_name}}}}}",
        rf"{{{note_text}}}",
        r"\end{figure}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX Appendix (appendix.tex) strictly from Gurobi .out logs.")
    parser.add_argument("--ablation-dir", type=str, default="results_N6_35",
                        help="Directory containing eta=0 ablation .out files (default: results_N6_35)")
    parser.add_argument("--pareto-dir", type=str, default="results_pareto",
                        help="Directory containing eta>0 Pareto .out files (default: results_pareto)")
    parser.add_argument("--output", type=str, default="appendix.tex",
                        help="Output LaTeX Appendix file path (default: appendix.tex)")
    args = parser.parse_args()

    common_note = (
        r"Blue blocks form the primary load-bearing support spine; orange blocks denote adjacent shield blocks; "
        r"teal blocks denote the top bridge block; purple blocks form the vertical counterweight towers."
    )

    stats_counter = {"requested": 0, "rendered": 0, "missing": []}

    # Section A specifications (eta = 0.00, N = 6..35)
    sec_a_groups = [
        ([6, 7, 8, 9, 10, 11],
         r"Certified Optimal Stack Configurations for $N \in \{6, 7, 8, 9, 10, 11\}$ Under $\eta = 0.00$",
         "fig:app_optimal_6_11",
         common_note + r" Numbers inside blocks indicate block indices $i \in \mathcal{N}$.",
         2.55, 0.22, "4pt"),
        ([12, 13, 14, 15, 16, 17],
         r"Balanced Stack Configurations in the Spinal Regime for $N \in \{12, 13, 14, 15, 16, 17\}$ Under $\eta = 0.00$",
         "fig:app_spinal_12_17",
         common_note, 3.45, 0.26, "10pt"),
        ([18, 19, 20, 21, 22, 23],
         r"Balanced Stack Configurations Across the Regime Transition for $N \in \{18, 19, 20, 21, 22, 23\}$ Under $\eta = 0.00$",
         "fig:app_transition_18_23",
         common_note, 3.45, 0.25, "10pt"),
        ([24, 25, 26, 27, 28, 29],
         r"Balanced Multi-Tower Stack Configurations for $N \in \{24, 25, 26, 27, 28, 29\}$ Under $\eta = 0.00$",
         "fig:app_dag_24_29",
         common_note, 3.45, 0.24, "10pt"),
        ([30, 31, 32, 33, 34, 35],
         r"Deep Multi-Tower Stack Configurations for $N \in \{30, 31, 32, 33, 34, 35\}$ Under $\eta = 0.00$",
         "fig:app_large_30_35",
         common_note, 3.45, 0.23, "10pt"),
    ]

    sec_a_figs = []
    for n_group, cap, lbl, note, th, bhm, rsep in sec_a_groups:
        specs = [
            (chr(ord("a") + idx), n_v, 0.00, os.path.join(args.ablation_dir, f"M4_WarmStart_N{n_v}.out"))
            for idx, n_v in enumerate(n_group)
        ]
        f_tex = build_grid_figure_from_logs(specs, cap, lbl, note, stats_counter, target_h=th, bh_max=bhm, row_sep=rsep)
        if f_tex is not None:
            sec_a_figs.append(f_tex)

    # Section B specifications (eta in {0.01, 0.02, 0.05, 0.10})
    def make_pareto_8panel_specs(n_first, n_second):
        specs = []
        idx_p = 0
        for n_v in [n_first, n_second]:
            for eta_v in [0.01, 0.02, 0.05, 0.10]:
                sub_lbl = chr(ord("a") + idx_p)
                fpath = os.path.join(args.pareto_dir, f"M4_WarmStart_N{n_v}_eta{eta_v:.2f}.out")
                specs.append((sub_lbl, n_v, eta_v, fpath))
                idx_p += 1
        return specs

    sec_b_groups = [
        (6, 7, r"Certified Optimal Stack Configurations Across $\eta \in \{0.01, 0.02, 0.05, 0.10\}$ for $N \in \{6, 7\}$",
         "fig:app_pareto_6_7", 2.15, 0.20, "4pt"),
        (8, 9, r"Certified Optimal Stack Configurations Across $\eta \in \{0.01, 0.02, 0.05, 0.10\}$ for $N \in \{8, 9\}$",
         "fig:app_pareto_8_9", 2.55, 0.22, "6pt"),
        (10, 11, r"Certified Optimal Stack Configurations Across $\eta \in \{0.01, 0.02, 0.05, 0.10\}$ for $N \in \{10, 11\}$",
         "fig:app_pareto_10_11", 2.55, 0.21, "6pt"),
        (12, 20, r"Balanced Stack Configurations Across $\eta \in \{0.01, 0.02, 0.05, 0.10\}$ for $N \in \{12, 20\}$",
         "fig:app_pareto_12_20", 2.55, 0.21, "6pt"),
        (30, 35, r"Balanced Stack Configurations Across $\eta \in \{0.01, 0.02, 0.05, 0.10\}$ for $N \in \{30, 35\}$",
         "fig:app_pareto_30_35", 2.55, 0.20, "6pt"),
    ]

    sec_b_figs = []
    for n1, n2, cap, lbl, th, bhm, rsep in sec_b_groups:
        specs = make_pareto_8panel_specs(n1, n2)
        f_tex = build_grid_figure_from_logs(specs, cap, lbl, common_note, stats_counter, target_h=th, bh_max=bhm, row_sep=rsep)
        if f_tex is not None:
            sec_b_figs.append(f_tex)

    if stats_counter["rendered"] == 0:
        print(f"[ERROR] No valid .out log files found in '{args.ablation_dir}' or '{args.pareto_dir}'.")
        print("        Please run `run_ablation.py` and `run_pareto.py` first before generating `appendix.tex`.")
        sys.exit(1)

    sec_a_body = "\n\n\\clearpage\n\n".join(sec_a_figs) if sec_a_figs else "% No ablation .out logs found."
    sec_b_body = "\n\n\\clearpage\n\n".join(sec_b_figs) if sec_b_figs else "% No Pareto .out logs found."

    appendix_content = f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Appendix for INFORMS Journal on Computing (IJOC)
%%  Auto-generated by generate_tikz_stacks.py from Gurobi .out logs
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\begin{{APPENDIX}}{{Visual Atlas of Balanced Overhang Stack Configurations}}
\\label{{app:VisualAtlas}}

This appendix presents a visual atlas of the balanced overhang stack configurations obtained by model $\\mathcal{{M}}_4$ across benchmark block counts $N \\in \\{{6, \\dots, 35\\}}$ under zero stability margin ($\\eta = 0.00$) and across positive relative contact stability margins $\\eta = \\epsilon / W \\in \\{{0.01, 0.02, 0.05, 0.10\\}}$. In all figures, blue blocks denote the primary load-bearing support spine transferring vertical loads to the table edge at $X = 0$, orange blocks denote adjacent shield blocks that apply stabilizing downward forces to the left edges of the spine blocks, teal blocks denote the top bridge block, and purple blocks denote vertical counterweight towers.

\\section{{Structural Evolution Across Block Counts Under Zero Stability Margin}}\\label{{app:ZeroMarginAtlas}}

{sec_a_body}

\\clearpage

\\section{{Structural Comparison Across Positive Stability Margins}}\\label{{app:ParetoAtlas}}

{sec_b_body}

\\end{{APPENDIX}}
"""

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(appendix_content)

    print(f"[SUCCESS] Rendered {stats_counter['rendered']}/{stats_counter['requested']} panels from existing .out logs to: {args.output}")
    if stats_counter["missing"]:
        print(f"[INFO] Skipped {len(stats_counter['missing'])} unrun panels (no .out file present on disk).")


if __name__ == "__main__":
    main()
