# MobileGym-Critic IRR

IRR stands for Information Retention Rate. It is an optional fine-grained metric for analyzing whether a GUI agent preserved and reused task-relevant information during a trajectory.

```text
IRR = correctly recalled and used information units / required information units
```

## Quick Start

Run all default checks:

```bash
cd mobilegym_critic/irr
python3 run_irr.py
```

Run individual utilities:

```bash
# Status check and demo
python3 demo_irr_functionality.py

# Status only
python3 demo_irr_functionality.py --status

# Detailed analysis for failed cases
python3 true_parallel_irr_processor.py --all
```

## Layout

```text
mobilegym_critic/irr/
|-- README.md
|-- IRR_IMPLEMENTATION_README.md
|-- run_irr.py
|-- demo_irr_functionality.py
|-- irr_agent.py
|-- quick_add_irr_columns.py
`-- true_parallel_irr_processor.py
```

## Metric Interpretation

- `IRR = 100%`: all required information was retained and used.
- `IRR = 50-99%`: part of the required information was lost or not used.
- `IRR = 0%`: information collection failed, or the agent did not retain required information.

## Features

- LLM-assisted IRR analysis for GUI trajectories.
- Parallel processing for large result sets.
- Progress saving and resumable batch execution.
- Non-destructive output: IRR columns are added without modifying the original task artifacts.

See [`IRR_IMPLEMENTATION_README.md`](IRR_IMPLEMENTATION_README.md) for implementation details.
