# MobileForge Pipeline

MobileForge connects five public code stages:

1. **Environment exploration**: `explore/` collects autonomous app exploration trajectories.
2. **MobileGym-Curriculum task generation**: `explore/curriculum_generator_refactored_add_aw_few-shot/` converts exploration traces into executable tasks.
3. **HiFPO rollout and MobileGym-Critic feedback**: `rollout/` runs multi-attempt hint-guided trajectories and evaluator feedback.
4. **Training-data construction**: `rollout/data_process_verl/` converts rollout sessions into step-level GRPO JSON.
5. **Step-level GRPO training**: `training/` trains Qwen3-VL and GUI-Owl-1.5 checkpoints with MobileForge rewards.

## Main Entry Points

| Stage | Entry point | Notes |
| --- | --- | --- |
| Exploration | `explore/exploration_and_mining.py` | Single-device DFS exploration over target Android apps. |
| Parallel exploration | `explore/interactive_parallel_exploration.py`, `explore/parallel_exploration/main.py` | Multi-device app exploration used to collect the released traces. |
| Curriculum generation | `explore/curriculum_generator_refactored_add_aw_few-shot/main.py` | Paper task-generation version with AndroidWorld few-shot support. |
| Rollout | `rollout/run.py` | Multi-attempt hint-guided rollout over generated tasks. |
| Critic/evaluation | `rollout/mobilegym_critic/evaluator.py` | Runs MobileGym-Critic trajectory outcome, process feedback, and corrective-hint evaluation. |
| Data processing | `rollout/data_process_verl/mobileforge_data_processor.py` | Converts rollout outputs to MobileForge GRPO JSON. |
| Training | `training/examples/qwen3_vl_8b_mobileforge_grpo.sh` | Public GRPO training entry. |
| AndroidWorld evaluation | `evaluation/androidworld/run.py`, `evaluation/androidworld/checkpoint_parser.py` | Reproduces AndroidWorld pass@k results. |
| MobileWorld evaluation | `evaluation/mobileworld/README.md` | Uses the official MobileWorld project and released trajectory logs. |

## Released Artifacts

- Training data: [🤗 `lgy0404/mobileforge-training-data`](https://huggingface.co/datasets/lgy0404/mobileforge-training-data)
- Generated tasks: [🤗 `lgy0404/mobileforge-generated-tasks`](https://huggingface.co/datasets/lgy0404/mobileforge-generated-tasks)
- Exploration trajectories: [🤗 `lgy0404/mobileforge-exploration-trajectories`](https://huggingface.co/datasets/lgy0404/mobileforge-exploration-trajectories)
- Benchmark logs/results: [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results)

Released training data, generated tasks, and exploration trajectories are grouped in the [🤗 MobileForge Datasets collection](https://huggingface.co/collections/lgy0404/mobileforge-datasets). Benchmark logs/results are released separately in [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results).
