# MobileForge Rollout Data Analyzer

`data_analyzer` builds an interactive HTML dashboard for MobileForge rollout sessions. It summarizes task, trajectory, step, hint, and filtering statistics, and helps inspect how different filtering strategies affect the training-data distribution.

## Layout

```text
data_analyzer/
|-- __init__.py
|-- __main__.py
|-- main.py            # CLI entry point
|-- loader.py          # Rollout data loader
|-- metrics.py         # Task/trajectory/step metrics
|-- filters.py         # Composable filtering strategies
|-- report.py          # HTML dashboard generator
|-- analysis_reports/  # Generated reports
`-- README.md
```

## Quick Start

Run from `rollout/`:

```bash
# Baseline analysis without filtering
python -m data_analyzer --rollout_dir /path/to/session-rollout

# Analysis with filtering options
python -m data_analyzer --rollout_dir /path/to/session-rollout \
  --remove_errors --remove_loops 3 --remove_infeasible 2 \
  --success_only --best_trajectory

# Multiple rollout directories
python -m data_analyzer --rollout_dir /dir1 /dir2 -o my_reports

# Debug on a subset
python -m data_analyzer --rollout_dir /path/to/session --max_tasks 20

# Analyze one app
python -m data_analyzer --rollout_dir /path/to/session --app_filter Clock
```

The tool writes a timestamped report directory under `data_analyzer/analysis_reports/`.

| File | Description |
| --- | --- |
| `dashboard.html` | Interactive dashboard; open it in a browser. |
| `analysis_data.json` | Raw analysis data for programmatic use. |

## Dashboard Tabs

- **Overview**: KPIs, impact distribution, trajectory success/failure, task SR, reasonableness, action types, feasibility, Pass@k, and hint coverage.
- **App Analysis**: per-app success rates, Pass@k curves, and detailed app-level metrics.
- **Filtering Experiments**: client-side controls for combining filtering strategies and comparing before/after distributions.
- **Precomputed Analysis**: independent effects of common filtering strategies.
- **Data Table**: searchable and sortable task-level table.

## Filtering Strategies

| # | Name | CLI option | Description |
| --- | --- | --- | --- |
| 1 | Best trajectory | `--best_trajectory` | Keep the best attempt per task. |
| 2 | Infeasible removal | `--remove_infeasible K` | Remove a task if infeasible votes are at least `K`. |
| 3 | SR range | `--sr_range MIN MAX` | Keep tasks with average SR in `[MIN, MAX]`. |
| 4 | Positive steps only | `--positive_only` | Remove steps whose impact is not positive. |
| 5 | Successful attempts only | `--success_only` | Remove failed attempts. |
| 6 | Loop removal | `--remove_loops K` | Remove attempts with at least `K` consecutive identical actions. |
| 7 | Evaluation errors | `--remove_errors` | Remove attempts with invalid final results. |
| 8 | Step-count range | `--step_range MIN MAX` | Keep attempts whose step count is in `[MIN, MAX]`. |

## Key Metrics

- **Task SR**: successful attempts divided by all attempts for a task.
- **Pass@k**: fraction of tasks solved at least once within the first `k` attempts.
- **Generated hints**: fraction of attempts with `eval_hint.json`.
- **Used hints**: fraction of attempts with `hints_input.json`.
- **Loop detection**: attempts with at least `k` consecutive identical actions in `log.json`.

## Dependencies

- Python 3.8 or newer.
- No required pip dependencies beyond the standard library.
- The dashboard uses Chart.js from a CDN, so internet access is needed when opening `dashboard.html`.
