# Benchmark Result Release

This document maps MobileForge paper results to the public AndroidWorld and MobileWorld archives in [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results). Training data, generated tasks, and exploration trajectories are grouped separately in the [🤗 MobileForge Datasets collection](https://huggingface.co/collections/lgy0404/mobileforge-datasets).

## Public Dataset Layout

```text
mobileforge-benchmark-results/
  README.md
  metadata/archives_manifest.json
  metadata/model_result_map.json
  archives/androidworld/<descriptive-experiment-name>.tar.zst
  archives/mobileworld/<descriptive-experiment-name>.tar.zst
```

## Result Mapping

| Experiment | Checkpoint | AndroidWorld public path | AndroidWorld result | MobileWorld public path | MobileWorld result |
| --- | --- | --- | --- | --- | --- |
| GUI-Owl-1.5-8B-Instruct base | `mPLUG/GUI-Owl-1.5-8B-Instruct` | [archives/androidworld/gui-owl-base.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/gui-owl-base.tar.zst) | P@1 65/116, P@2 79/116, P@3 80/116 | [archives/mobileworld/gui-owl-base.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/mobileworld/gui-owl-base.tar.zst) | SR 44/117 |
| GUI-Owl-1.5-8B + MobileForge 200 tasks | `lgy0404/ForgeOwl-8B-200tasks` | [archives/androidworld/gui-owl-mobileforge-200tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/gui-owl-mobileforge-200tasks.tar.zst) | P@1 75/116, P@2 85/116, P@3 86/116 | - | - |
| GUI-Owl-1.5-8B + MobileForge 400 tasks | `lgy0404/ForgeOwl-8B-400tasks` | [archives/androidworld/gui-owl-mobileforge-400tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/gui-owl-mobileforge-400tasks.tar.zst) | P@1 75/116, P@2 86/116, P@3 90/116 | - | - |
| GUI-Owl-1.5-8B + MobileForge 900 tasks | `lgy0404/ForgeOwl-8B` | [archives/androidworld/gui-owl-mobileforge-900tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/gui-owl-mobileforge-900tasks.tar.zst) | P@1 78/116, P@2 87/116, P@3 90/116 | [archives/mobileworld/gui-owl-mobileforge-900tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/mobileworld/gui-owl-mobileforge-900tasks.tar.zst) | SR 48/117 |
| Qwen3-VL-8B-Instruct base | `Qwen/Qwen3-VL-8B-Instruct` | [archives/androidworld/qwen3-vl-base.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/qwen3-vl-base.tar.zst) | P@1 47/116, P@2 57/116, P@3 64/116 | [archives/mobileworld/qwen3-vl-base.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/mobileworld/qwen3-vl-base.tar.zst) | SR 9/117 |
| Qwen3-VL-8B + MobileForge 200 tasks | `lgy0404/ForgeQwen3-8B-200tasks` | [archives/androidworld/qwen3-vl-mobileforge-200tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/qwen3-vl-mobileforge-200tasks.tar.zst) | P@1 55/116, P@2 64/116, P@3 71/116 | - | - |
| Qwen3-VL-8B + MobileForge 400 tasks | `lgy0404/ForgeQwen3-8B-400tasks` | [archives/androidworld/qwen3-vl-mobileforge-400tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/qwen3-vl-mobileforge-400tasks.tar.zst) | P@1 61/116, P@2 69/116, P@3 73/116 | - | - |
| Qwen3-VL-8B + MobileForge 900 tasks | `lgy0404/ForgeQwen3-8B` | [archives/androidworld/qwen3-vl-mobileforge-900tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/androidworld/qwen3-vl-mobileforge-900tasks.tar.zst) | P@1 59/116, P@2 70/116, P@3 78/116 | [archives/mobileworld/qwen3-vl-mobileforge-900tasks.tar.zst](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results/blob/main/archives/mobileworld/qwen3-vl-mobileforge-900tasks.tar.zst) | SR 12/117 |

## Reproduction Notes

AndroidWorld evaluation code is included under `evaluation/androidworld/` and exposes the Qwen3-VL and GUI-Owl agents used by the paper. After configuring `config.yaml`, representative commands are:

```bash
python -u run.py --checkpoint_dir ./results/mobileforge-aw-qwen3-900 --agent_name Qwen3VL --n_task_combinations=3
python -u run.py --checkpoint_dir ./results/mobileforge-aw-guiowl-900 --agent_name GUIOwl15 --n_task_combinations=3
```

AndroidWorld summaries can be parsed with `checkpoint_parser.py`.

MobileWorld evaluation follows the upstream MobileWorld workflow. Logs can be inspected with:

```bash
uv run mw logs view --log_dir traj_logs/qwen3_vl_logs
```
