# Polyhedral Strengthening and Global Optimization for the Single-Root Maximum Overhang Problem

This repository provides the Mixed-Integer Bilinear Programming (MIBLP) formulations, polyhedral strengthening modules, analytical-inversion warm-start constructors, batch evaluation scripts, and raw Gurobi solver logs for the paper:

> **Polyhedral Strengthening and Global Optimization for the Single-Root Maximum Overhang Problem**

---

## 1. Overview of Formulations and Cumulative Reformulation Stages

The maximum overhang problem seeks the maximum horizontal distance from the edge of a flat table ($X = 0$) to the rightmost edge of any block in a balanced stack of $N$ identical homogeneous frictionless blocks of unit length ($2W = 1$) and unit weight ($m = 1$) over single-root Directed Acyclic Graph (DAG) support topologies.

To evaluate the computational contribution of each mathematical component, the codebase implements the baseline MIBLP formulation (**M<sub>0</sub>**) and four cumulative reformulation stages (**M<sub>1</sub>** through **M<sub>4</sub>**):

| Model | Script / Identifier | Incorporated Mathematical and Algorithmic Components |
| :--- | :--- | :--- |
| **M<sub>0</sub>** | `Baseline` (`maxOverhang_baseline.py`) | Global coordinate MIBLP formulation with unique table-root connectivity (<i>L</i><sub>1</sub> = 1, <i>T</i><sub>1,0</sub> = 1) and big-<i>M</i> indicator constraints. Provides unconditional optimality certifications over general single-root DAGs for <i>N</i> &le; 11. |
| **M<sub>1</sub>** | `M1_OrderPrune` (`ablation_m1_order_prune.py`) | **M<sub>0</sub>** + upper-DAG variable elimination (1 &le; <i>j</i> &lt; <i>i</i>), two-support degree bound (&sum;<sub><i>j</i>=1</sub><sup><i>i</i>&minus;1</sup> <i>T</i><sub><i>i,j</i></sub> &le; 2) and layer-indexed force capacities <i>u</i><sub><i>i</i></sub><sup><i>F</i></sup> = (<i>N</i> &minus; <i>i</i> + 1)<i>m</i>, and base saddle linearization &tau;<sub>1,0</sub> = <i>NmP</i><sub>1,0</sub>. |
| **M<sub>2</sub>** | `M2_LocalMoment` (`ablation_m2_local_moment.py`) | **M<sub>1</sub>** + conditional sub-stack overhang bound <i>D</i><sup>*</sup>(<i>N</i> &minus; 1) for <i>N</i> &ge; 14 (using unconditional physical bound <i>NW</i> + <i>W</i> for <i>N</i> &le; 13), block-indexed reachability intervals [<i>p</i><sub><i>i,j</i></sub><sup>lb</sup>, <i>p</i><sub><i>i,j</i></sub><sup>ub</sup>], tight disjunctive contact-interval inequalities, and global table-torque identity &sum;<sub><i>i</i>=1</sub><sup><i>N</i></sup> <i>mX</i><sub><i>i</i></sub> = &tau;<sub>1,0</sub>. |
| **M<sub>3</sub>** | `M3_ConvexHull` (`ablation_m3_convex_hull.py`) | **M<sub>2</sub>** + macro-block overturning envelopes, tightened base coordinate bound <i>X</i><sub>1</sub> &le; ((<i>N</i> &minus; 1)/<i>N</i>)<i>W</i> and implied linear base-torque bounds, and sparsity-preserving layer-jump and adjacent-support locking cuts. |
| **M<sub>4</sub>** | `M4_WarmStart` (`ablation_m4_warm_start.py`) | **M<sub>3</sub>** + indicator-lifted 4D McCormick tetrahedral facets and recursive sub-stack cuts for <i>N</i> &ge; 10, analytical-inversion multi-tower shielded spinal MIP start, and spinal-regime minimum stack height pruning <i>L</i><sub><i>N</i></sub> &ge; <i>L</i><sub><i>N</i></sub><sup>*</sup>. |

All models support both the zero-margin equilibrium regime (&eta; = &epsilon; / <i>W</i> = 0.00) and positive contact stability margins (&eta; = &epsilon; / <i>W</i> &gt; 0).

---

## 2. Repository Structure

```text
.
├── maxOverhang_baseline.py      # Standalone solver for the baseline MIBLP model M0 (Baseline)
├── maxOverhang_proposed.py      # Main CLI dispatcher and shared callback/logging utilities for M1-M4
├── ablation_m1_order_prune.py   # Model M1: Upper-DAG reduction and base saddle linearization
├── ablation_m2_local_moment.py  # Model M2: Sub-stack bounds and tight disjunctive contact inequalities
├── ablation_m3_convex_hull.py   # Model M3: Macro-block envelopes, base-torque bounds, and layer cuts
├── ablation_m4_warm_start.py    # Model M4: 4D McCormick facets, warm-start, and spinal layer pruning
├── run_ablation.py              # Parallel batch runner for the 5-model comparison (N = 6..35)
├── run_pareto.py                # Parallel batch runner for the stability-margin trade-off study (M4)
├── parse_results.py             # Log parser aggregating .out files into summary CSV reports
├── generate_tikz_stacks.py      # Automated LaTeX TikZ vector stack atlas generator (appendix.tex)
├── requirements.txt             # Python package dependencies
├── LICENSE                      # MIT License
└── README.md                    # Documentation and reproduction instructions
```

> **Note on Output Directory Isolation:** All experiment runners and single-instance scripts write `.out` solver logs, `.png` figures, and `.csv` statistical summaries strictly inside the target output directory specified via `--outdir` (`results_N6_35/` or `results_pareto/`).

---

## 3. Environment and Dependency Requirements

The codebase is platform-independent and runs on **Linux, macOS, and Windows** wherever Python and Gurobi Optimizer are installed.

### Software Requirements
* **Python:** 3.9 or higher
* **Gurobi Optimizer:** Version **13.0.3**
* **Python Packages:** `gurobipy==13.0.3`, `matplotlib>=3.7.0`

### Installation via `pip`
```bash
pip install -r requirements.txt
```

Reference configuration used in the paper: Intel Core Ultra 7 265K processor, 32 GB RAM, Ubuntu 24.04 LTS, Gurobi Optimizer 13.0.3, single thread `Threads = 1`, fixed random seed `Seed = 42`, relative optimality gap tolerance `MIPGap = 5e-4` ($\tau_{\text{gap}} = 0.05\%$), wall-clock time limit `TimeLimit = 3600` seconds.

---

## 4. Usage of Individual Scripts

### 4.1 Solving a Single Instance with Models M<sub>1</sub>&ndash;M<sub>4</sub> (`maxOverhang_proposed.py`)
```bash
# Solve N = 11 with M4_WarmStart, 3600s time limit, 1 thread, saving log to results_test/M4_WarmStart_N11.out
python maxOverhang_proposed.py 11 3600 --method M4_WarmStart --threads 1 --gap 5e-4 --seed 42 --outdir results_test
```

* To solve under a positive contact stability margin (&eta; = &epsilon; / <i>W</i> = 0.05):
  ```bash
  python maxOverhang_proposed.py 12 3600 --method M4_WarmStart --eta 0.05 --threads 1 --outdir results_pareto
  ```
* To render a `.png` visualization of the resulting stack inside `--outdir`, append `--plot`.

### 4.2 Solving a Single Instance with Baseline Model M<sub>0</sub> (`maxOverhang_baseline.py`)
```bash
# Solve N = 10 with Baseline (M0), 3600s time limit, saving log to results_test/Baseline_N10.out
python maxOverhang_baseline.py 10 3600 --threads 1 --gap 5e-4 --seed 42 --outdir results_test
```

### 4.3 Parsing `.out` Logs into Summary CSVs (`parse_results.py`)
`parse_results.py` scans all `.out` files inside `--indir` and writes the aggregated summary table (`<indir>/ablation_summary.csv` by default).
```bash
# Aggregate all .out logs in results_N6_35/ into results_N6_35/ablation_summary.csv
python parse_results.py --indir results_N6_35
```
* **Optional Time-Series Trace Export (`--save-trace`):** By default, time-series progress traces (`convergence_trace.csv` / `pareto_trace.csv`) are not generated to avoid large file sizes during multi-hour runs. To export traces, pass `--save-trace`:
  ```bash
  python parse_results.py --indir results_N6_35 --save-trace
  ```

### 4.4 Generating the LaTeX TikZ Visual Stack Atlas (`generate_tikz_stacks.py`)
`generate_tikz_stacks.py` reads the `.out` log files present in `--ablation-dir` and `--pareto-dir` and renders the corresponding TikZ stack figures with 5-decimal overhang labels into `appendix.tex`:
```bash
python generate_tikz_stacks.py --ablation-dir results_N6_35 --pareto-dir results_pareto --output appendix.tex
```

---

## 5. Reproducing the Paper's Experimental Results

### 5.1 Reproducing the Five-Stage Comparison (Tables 2 & 3: M<sub>0</sub>&ndash;M<sub>4</sub>, <i>N</i> = 6, &hellip;, 35)

To run all 5 models across all 30 instances (<i>N</i> = 6, &hellip;, 35; 150 runs total) with a 3600-second time limit per instance, 1 solver thread per run (`-t 1`), and 5 concurrent worker processes (`-j 5`):

```bash
nohup python -u run_ablation.py \
    --n-list $(seq 6 35) \
    --methods Baseline M1_OrderPrune M2_LocalMoment M3_ConvexHull M4_WarmStart \
    --time-limit 3600 \
    -j 5 -t 1 \
    --gap 5e-4 --seed 42 \
    --outdir results_N6_35 > master_ablation.log 2>&1 &
```

* **Fast reproduction of the solved regime (Table 2: <i>N</i> = 6, &hellip;, 11; ~45 minutes with `-j 5`):**
  ```bash
  python -u run_ablation.py --n-list 6 7 8 9 10 11 --time-limit 3600 -j 5 -t 1 --outdir results_N6_35
  ```
* **Output Location:** All `.out` logs and the aggregated table `results_N6_35/ablation_summary.csv` are saved inside `results_N6_35/`.

### 5.2 Reproducing the Stability-Margin Trade-Off Study (Table 4: M<sub>4</sub>, &eta; &in; {0.01, 0.02, 0.05, 0.10})

To run Model **M<sub>4</sub>** (`M4_WarmStart`) across the four positive relative stability margins &eta; &in; {0.01, 0.02, 0.05, 0.10} and all 13 representative block counts <i>N</i> &in; {6, 7, 8, 9, 10, 11, 12, 15, 19, 20, 25, 30, 35} (52 runs total):

```bash
nohup python -u run_pareto.py \
    --n-list 6 7 8 9 10 11 12 15 19 20 25 30 35 \
    --eta-list 0.01 0.02 0.05 0.10 \
    --time-limit 3600 \
    -j 5 -t 1 \
    --gap 5e-4 --seed 42 \
    --outdir results_pareto > master_pareto.log 2>&1 &
```

* **Fast reproduction of the spinal-regime certified positive-margin instances (<i>N</i> = 6, &hellip;, 11; ~15 minutes with `-j 5`):**
  ```bash
  python -u run_pareto.py --n-list 6 7 8 9 10 11 --eta-list 0.01 0.02 0.05 0.10 --time-limit 3600 -j 5 -t 1 --outdir results_pareto
  ```
* **Output Location:** All `.out` logs and the aggregated table `results_pareto/pareto_summary.csv` are saved inside `results_pareto/`.

---

## 6. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
