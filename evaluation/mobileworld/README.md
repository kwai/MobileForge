# MobileForge MobileWorld Evaluation

MobileForge uses the official MobileWorld framework for MobileWorld evaluation. See https://github.com/Tongyi-MAI/MobileWorld for environment setup and task execution.

## Agent Mapping

- `gui_owl_1_5`: use with `mPLUG/GUI-Owl-1.5-8B-Instruct` and `lgy0404/ForgeOwl-8B`.
- `qwen3vl`: use with `Qwen/Qwen3-VL-8B-Instruct` and `lgy0404/ForgeQwen3-*` checkpoints.

## Viewing Logs

After running MobileWorld or after downloading the released benchmark-result dataset, inspect trajectories with the MobileWorld log viewer:

```bash
uv run mw logs view --log_dir traj_logs/qwen3_vl_logs
```

The MobileWorld result archives used in the paper are released in [🤗 `lgy0404/mobileforge-benchmark-results`](https://huggingface.co/datasets/lgy0404/mobileforge-benchmark-results). See [`../../docs/evaluation_results.md`](../../docs/evaluation_results.md) for the MobileWorld archive mapping.
