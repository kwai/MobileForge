# MobileForge Models

The MobileForge model checkpoints are hosted in the Hugging Face collection:

- [🤗 MobileForge Models collection](https://huggingface.co/collections/lgy0404/mobileforge-models)

| Model ID | Base model family | Training tasks | Paper use |
| --- | --- | --- | --- |
| `lgy0404/ForgeQwen3-8B` | Qwen3-VL-8B-Instruct | 900 | Main Qwen3-VL MobileForge checkpoint. |
| `lgy0404/ForgeQwen3-8B-400tasks` | Qwen3-VL-8B-Instruct | 400 | Scaling ablation. |
| `lgy0404/ForgeQwen3-8B-200tasks` | Qwen3-VL-8B-Instruct | 200 | Scaling ablation. |
| `lgy0404/ForgeOwl-8B` | GUI-Owl-1.5-8B-Instruct | 900 | Main GUI-Owl MobileForge checkpoint. |
| `lgy0404/ForgeOwl-8B-400tasks` | GUI-Owl-1.5-8B-Instruct | 400 | Scaling ablation. |
| `lgy0404/ForgeOwl-8B-200tasks` | GUI-Owl-1.5-8B-Instruct | 200 | Scaling ablation. |

MobileWorld agent mapping:

- ForgeOwl checkpoints use the `gui_owl_1_5` agent path.
- ForgeQwen3 checkpoints use the `qwen3vl` agent path.

AndroidWorld evaluation uses the Qwen3-VL and GUI-Owl agents provided in `evaluation/androidworld/`.
