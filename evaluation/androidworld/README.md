# MobileForge AndroidWorld Evaluation

This directory contains the AndroidWorld evaluation fork used for the MobileForge paper. The public release keeps the Qwen3-VL and GUI-Owl-1.5 agents relevant to the reported results.

## Agents

- `android_world/agents/qwen3_vl.py`: Qwen3-VL style mobile GUI agent. Use this for `Qwen/Qwen3-VL-8B-Instruct` and the released `lgy0404/ForgeQwen3-*` checkpoints.
- `android_world/agents/gui_owl.py`: GUI-Owl-1.5 style agent. Use this for `mPLUG/GUI-Owl-1.5-8B-Instruct` and the released `lgy0404/ForgeOwl-*` checkpoints.

## Configuration

Copy `config.yaml.example` to `config.yaml` and fill in your model endpoint, key, and model id. The released checkpoints are hosted at:

- `lgy0404/ForgeQwen3-8B`
- `lgy0404/ForgeQwen3-8B-400tasks`
- `lgy0404/ForgeQwen3-8B-200tasks`
- `lgy0404/ForgeOwl-8B`
- `lgy0404/ForgeOwl-8B-400tasks`
- `lgy0404/ForgeOwl-8B-200tasks`

## Run Examples

```bash
python -u run.py   --checkpoint_dir './results/mobileforge-aw-qwen3vl'   --agent_name 'Qwen3VL'   --n_task_combinations=3
```

```bash
python -u run.py   --checkpoint_dir './results/mobileforge-aw-guiowl15'   --agent_name 'GUIOwl15'   --n_task_combinations=3
```

After evaluation, parse pass@k results with:

```bash
python checkpoint_parser.py --checkpoint_dir './results/mobileforge-aw-qwen3vl'
```

The paper result logs are released in [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results). See [`../../docs/evaluation_results.md`](../../docs/evaluation_results.md) for the AndroidWorld archive mapping.
