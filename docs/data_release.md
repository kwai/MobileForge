# MobileForge Data Release

This document records the public artifacts released for MobileForge. Training data, exploration trajectories, and generated tasks are grouped in the [🤗 MobileForge Datasets collection](https://huggingface.co/collections/lgy0404/mobileforge-datasets). Benchmark archives are released separately at [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results).

| Artifact | Hugging Face dataset | Contents |
| --- | --- | --- |
| Training and validation data | [🤗 `lgy0404/mobileforge-training-data`](https://huggingface.co/datasets/lgy0404/mobileforge-training-data) | Six MobileForge GRPO training splits and one validation set. |
| Main benchmark results | [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results) | AndroidWorld and MobileWorld evaluation archives with model-to-result mappings. |
| Exploration trajectories | [🤗 `lgy0404/mobileforge-exploration-trajectories`](https://huggingface.co/datasets/lgy0404/mobileforge-exploration-trajectories) | Raw and parsed/final exploration trajectories for 20 Android apps. |
| Generated tasks | [🤗 `lgy0404/mobileforge-generated-tasks`](https://huggingface.co/datasets/lgy0404/mobileforge-generated-tasks) | Consolidated MobileGym-Curriculum task CSV. |

## Training Data

| Split | File | Description |
| --- | --- | --- |
| `qwen3-vl-900tasks` | `mobileforge_grpo_20260313_081857_tasks_900.json` | Qwen3-VL-8B MobileForge 900-task training split. |
| `qwen3-vl-400tasks` | `mobileforge_grpo_20260313_081857_tasks_400.json` | Qwen3-VL-8B scaling split. |
| `qwen3-vl-200tasks` | `mobileforge_grpo_20260313_081857_tasks_200.json` | Qwen3-VL-8B scaling split. |
| `gui-owl-900tasks` | `mobileforge_grpo_20260508_102018_tasks_900.json` | GUI-Owl-1.5-8B MobileForge 900-task training split. |
| `gui-owl-400tasks` | `mobileforge_grpo_20260508_102018_tasks_400.json` | GUI-Owl-1.5-8B scaling split. |
| `gui-owl-200tasks` | `mobileforge_grpo_20260329_093821_tasks_200.json` | GUI-Owl-1.5-8B scaling split. |
| `validation` | `mobileforge_grpo_20260307_021142_h3.json` | Validation set used during GRPO training. |

## Benchmark Results

The benchmark dataset uses compressed public archives such as `archives/androidworld/qwen3-vl-mobileforge-900tasks.tar.zst` and provides archive metadata in `metadata/model_result_map.json`. See `docs/evaluation_results.md` for the model-to-result mapping.

AndroidWorld results can be parsed with `evaluation/androidworld/checkpoint_parser.py`. MobileWorld logs can be inspected with the upstream MobileWorld command:

```bash
uv run mw logs view --log_dir <log_dir>
```

## Exploration Trajectories and Generated Tasks

The [🤗 `mobileforge-exploration-trajectories`](https://huggingface.co/datasets/lgy0404/mobileforge-exploration-trajectories) dataset contains:

- `raw`: original parallel exploration outputs.
- `final`: parsed/final outputs used by MobileGym-Curriculum.

The [🤗 `generated_tasks_26020301-all.csv`](https://huggingface.co/datasets/lgy0404/mobileforge-generated-tasks/blob/main/generated_tasks_26020301-all.csv) file contains 3,249 generated tasks with task description, app metadata, golden steps, difficulty level, and trajectory provenance.

## Utility Scripts

The training code includes two optional preprocessing helpers:

- `training/tools/create_scaling_splits.py`: creates 200/400/900-task scaling splits for ablation studies.
- `training/tools/extract_images_to_files.py`: extracts embedded images to files to reduce memory pressure when loading large JSON datasets.

The released JSON files in [🤗 `mobileforge-training-data`](https://huggingface.co/datasets/lgy0404/mobileforge-training-data) are already prepared, so these helpers are not required for standard reproduction.
