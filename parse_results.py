#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_results.py
Log Parser and Statistical CSV Aggregator for Maximum Overhang MIBLP Experiments.

Reads `.out` solver logs from `--indir` and writes the summary CSV directly inside `--indir`.
Exporting the time-series progress trace CSV (`convergence_trace.csv` or `pareto_trace.csv`)
is optional (`--save-trace`) and disabled by default to conserve memory and disk space.
"""

import os
import re
import glob
import csv
import math
import argparse


def parse_single_out_file(filepath, collect_trace=False):
    """Parse a single .out log file and return its summary dictionary and optional progress trace list."""
    filename = os.path.basename(filepath)
    if "_tree_" in filename.lower():
        return None, []

    record = {
        "Method": "Unknown",
        "N": None,
        "Eta": 0.0,
        "Status": "UNKNOWN",
        "ObjVal": None,
        "BestBound": None,
        "MIPGap_Pct": None,
        "Runtime_Sec": None,
        "FirstSol_Sec": None,
        "NodeCount": None,
        "SolCount": None,
    }
    trace = []

    m_fname = re.match(r"([A-Za-z0-9_]+?)(?:_dag)?_N(\d+)(?:_eta([0-9.]+))?(?:_s\d+)?\.out", filename, re.IGNORECASE)
    if m_fname:
        record["Method"] = m_fname.group(1)
        record["N"] = int(m_fname.group(2))
        if m_fname.group(3) is not None:
            record["Eta"] = float(m_fname.group(3))

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    in_summary = False
    for line in lines:
        s = line.strip()

        if "MODE = TREE" in s.upper() or "MODE=TREE" in s.upper():
            return None, []

        if s.startswith("META:"):
            m_method = re.search(r"Method=([^\s|]+)", s)
            m_n = re.search(r"N=(\d+)", s)
            m_eta = re.search(r"Eta=([0-9.]+)", s)
            if m_method:
                record["Method"] = m_method.group(1)
            if m_n:
                record["N"] = int(m_n.group(1))
            if m_eta:
                record["Eta"] = float(m_eta.group(1))

        elif s.startswith("--- EXPERIMENT START"):
            m_leg = re.search(r"N\s*=\s*(\d+)", s)
            m_eta_start = re.search(r"ETA\s*=\s*([0-9.]+)", s)
            if m_leg and record["N"] is None:
                record["N"] = int(m_leg.group(1))
            if m_eta_start:
                record["Eta"] = float(m_eta_start.group(1))

        elif collect_trace and s.startswith("[PROGRESS]"):
            m_prog = re.search(
                r"Time:\s*([0-9.]+)s\s*\|\s*Event:\s*([A-Z_]+)\s*\|\s*"
                r"ObjVal:\s*([0-9.NA+-]+)\s*\|\s*BestBound:\s*([0-9.NA+-]+)\s*\|\s*"
                r"Gap:\s*([0-9.NA+-]+)%?\s*\|\s*Nodes:\s*(\d+)",
                s
            )
            if m_prog:
                t_val = float(m_prog.group(1))
                ev = m_prog.group(2)
                obj_s = m_prog.group(3)
                bnd_s = m_prog.group(4)
                gap_s = m_prog.group(5)
                nds = int(m_prog.group(6))

                obj_v = float(obj_s) if obj_s != "NA" else None
                bnd_v = float(bnd_s) if bnd_s != "NA" else None
                gap_v = float(gap_s) if gap_s != "NA" else None

                trace.append({
                    "Method": record["Method"],
                    "N": record["N"],
                    "Eta": record["Eta"],
                    "Time_Sec": t_val,
                    "Event": ev,
                    "ObjVal": obj_v,
                    "BestBound": bnd_v,
                    "MIPGap_Pct": gap_v,
                    "Nodes": nds
                })

        elif s.startswith("SUMMARY_REPORT:"):
            in_summary = True
            continue
        elif in_summary:
            if s.startswith("===="):
                in_summary = False
                continue
            if ":" in s:
                k, v = [x.strip() for x in s.split(":", 1)]
                if k == "Method":
                    record["Method"] = v
                elif k == "N":
                    record["N"] = int(v)
                elif k == "Eta":
                    record["Eta"] = float(v)
                elif k == "Status":
                    st = v.upper()
                    if "TIMEOUT" in st or "TIME_LIMIT" in st or "TIME LIMIT" in st:
                        record["Status"] = "TIMEOUT"
                    else:
                        record["Status"] = st
                elif k == "ObjVal":
                    record["ObjVal"] = float(v) if v.lower() != "nan" else None
                elif k == "BestBound":
                    record["BestBound"] = float(v) if v.lower() != "nan" else None
                elif k == "MIPGap_Pct":
                    v_clean = v.replace("%", "").strip()
                    record["MIPGap_Pct"] = float(v_clean) if v_clean.lower() != "nan" else None
                elif k == "Runtime_Sec":
                    record["Runtime_Sec"] = float(v) if v.lower() != "nan" else None
                elif k == "FirstSol_Sec":
                    record["FirstSol_Sec"] = float(v) if v.lower() != "nan" else None
                elif k == "NodeCount":
                    record["NodeCount"] = int(float(v))
                elif k == "SolCount":
                    record["SolCount"] = int(float(v))

        if "Time limit reached" in s or "TIME LIMIT REACHED" in s:
            record["Status"] = "TIMEOUT"
        elif "Optimal solution found" in s or "Termination Status: OPTIMAL" in s:
            if record["Status"] == "UNKNOWN":
                record["Status"] = "OPTIMAL"
        elif s.startswith("Best objective"):
            m_gurobi = re.search(
                r"Best objective\s+([0-9.eE+-]+|-\s*),\s+best bound\s+([0-9.eE+-]+|-\s*),\s+gap\s+([0-9.eE+-]+|-)",
                s
            )
            if m_gurobi:
                if record["ObjVal"] is None and m_gurobi.group(1).strip() != "-":
                    record["ObjVal"] = float(m_gurobi.group(1))
                if record["BestBound"] is None and m_gurobi.group(2).strip() != "-":
                    record["BestBound"] = float(m_gurobi.group(2))
                if record["MIPGap_Pct"] is None and m_gurobi.group(3).strip() != "-":
                    record["MIPGap_Pct"] = float(m_gurobi.group(3))
        elif s.startswith("Explored ") and "nodes" in s:
            m_exp = re.search(r"Explored\s+(\d+)\s+nodes.*in\s+([0-9.]+)\s+seconds", s)
            if m_exp:
                if record["NodeCount"] is None:
                    record["NodeCount"] = int(m_exp.group(1))
                if record["Runtime_Sec"] is None:
                    record["Runtime_Sec"] = float(m_exp.group(2))

    return record, trace


def fmt_val(val, decimals=5):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NA"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Parse Overhang Tower .out logs and export summary CSVs")
    parser.add_argument("--indir", type=str, default="results",
                        help="Directory containing .out log files (default: results)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output summary CSV file path (default: <indir>/ablation_summary.csv)")
    parser.add_argument("--save-trace", action="store_true", default=False,
                        help="Optionally export time-series progress trace CSV for debugging (default: False)")
    parser.add_argument("--trace-output", type=str, default=None,
                        help="Output trace CSV file path when --save-trace is enabled (default: <indir>/convergence_trace.csv)")
    parser.add_argument("--include-eta", action="store_true", default=False,
                        help="Include Eta column in output CSV (automatically enabled if any Eta > 0 is present)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.indir, "ablation_summary.csv")
    if args.trace_output is None:
        args.trace_output = os.path.join(args.indir, "convergence_trace.csv")

    out_files = sorted(glob.glob(os.path.join(args.indir, "*.out")))
    if not out_files:
        print(f"[WARN] No .out files found in directory '{args.indir}'.")
        return

    records = []
    all_traces = []
    has_nonzero_eta = args.include_eta
    for fpath in out_files:
        rec, tr = parse_single_out_file(fpath, collect_trace=args.save_trace)
        if rec is not None:
            if rec.get("Eta", 0.0) > 0.0:
                has_nonzero_eta = True
            records.append(rec)
            if args.save_trace:
                all_traces.extend(tr)

    method_order = {
        "Baseline": 0,
        "M1_OrderPrune": 1,
        "M2_LocalMoment": 2,
        "M3_ConvexHull": 3,
        "M4_WarmStart": 4,
    }
    records.sort(key=lambda r: (
        r["N"] if r["N"] is not None else 9999,
        round(r.get("Eta", 0.0), 6),
        method_order.get(r["Method"], 99),
        r["Method"] or ""
    ))

    if has_nonzero_eta:
        fieldnames = [
            "Method", "N", "Eta", "Status",
            "ObjVal", "BestBound", "MIPGap_Pct",
            "Runtime_Sec", "FirstSol_Sec", "NodeCount", "SolCount"
        ]
    else:
        fieldnames = [
            "Method", "N", "Status",
            "ObjVal", "BestBound", "MIPGap_Pct",
            "Runtime_Sec", "FirstSol_Sec", "NodeCount", "SolCount"
        ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "Method": r["Method"],
                "N": r["N"] if r["N"] is not None else "NA",
                "Status": r["Status"],
                "ObjVal": fmt_val(r["ObjVal"], 5),
                "BestBound": fmt_val(r["BestBound"], 5),
                "MIPGap_Pct": fmt_val(r["MIPGap_Pct"], 4),
                "Runtime_Sec": fmt_val(r["Runtime_Sec"], 4),
                "FirstSol_Sec": fmt_val(r["FirstSol_Sec"], 4),
                "NodeCount": r["NodeCount"] if r["NodeCount"] is not None else "NA",
                "SolCount": r["SolCount"] if r["SolCount"] is not None else "NA",
            }
            if has_nonzero_eta:
                row["Eta"] = fmt_val(r.get("Eta", 0.0), 4)
            writer.writerow(row)

    print(f"[SUCCESS] Wrote {len(records)} experiment summary rows to: {args.output}")

    if args.save_trace and all_traces:
        if has_nonzero_eta:
            trace_fields = [
                "Method", "N", "Eta", "Time_Sec", "Event",
                "ObjVal", "BestBound", "MIPGap_Pct", "Nodes"
            ]
        else:
            trace_fields = [
                "Method", "N", "Time_Sec", "Event",
                "ObjVal", "BestBound", "MIPGap_Pct", "Nodes"
            ]
        os.makedirs(os.path.dirname(os.path.abspath(args.trace_output)), exist_ok=True)
        with open(args.trace_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trace_fields)
            writer.writeheader()
            for t in all_traces:
                t_row = {
                    "Method": t["Method"],
                    "N": t["N"],
                    "Time_Sec": fmt_val(t["Time_Sec"], 2),
                    "Event": t["Event"],
                    "ObjVal": fmt_val(t["ObjVal"], 5),
                    "BestBound": fmt_val(t["BestBound"], 5),
                    "MIPGap_Pct": fmt_val(t["MIPGap_Pct"], 4),
                    "Nodes": t["Nodes"]
                }
                if has_nonzero_eta:
                    t_row["Eta"] = fmt_val(t.get("Eta", 0.0), 4)
                writer.writerow(t_row)
        print(f"[SUCCESS] Wrote {len(all_traces)} progress trace rows to: {args.trace_output}")


if __name__ == "__main__":
    main()
