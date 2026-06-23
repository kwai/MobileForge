# MobileForge Training Tools

This directory contains optional preprocessing helpers for MobileForge training data.

## `create_scaling_splits.py`

Creates nested task-count splits for scaling ablations, such as 200/400/900 tasks. The released training JSON files are already prepared, so this script is only needed when rebuilding ablation splits from a larger GRPO dataset.

```bash
python tools/create_scaling_splits.py   --data_path data/mobileforge_grpo_full.json   --output_dir data/scaling_splits   --task_counts 200 400 900   --seed 42
```

## `extract_images_to_files.py`

Streams a large GRPO JSON, writes embedded base64 images to files, and replaces image payloads with file paths. Use it when a JSON dataset is too large to load comfortably in memory.

```bash
pip install ijson
python tools/extract_images_to_files.py   --input data/mobileforge_grpo.json   --output data/mobileforge_grpo_image_paths.json   --image_dir data/mobileforge_grpo_images   --path_mode relative
```
