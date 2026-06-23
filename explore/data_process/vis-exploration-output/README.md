# Exploration Output Visualizer

This tool parses and visualizes MobileForge exploration outputs. It is designed for large exploration datasets and includes memory-aware batch processing.

## Features

- Batch processing for large trajectory folders.
- Screenshot extraction and organization by app, trajectory, and step.
- App-level and trajectory-level statistics.
- JSON summaries for downstream processing.
- Optional lightweight mode for quick inspection.
- Memory monitoring and cleanup controls.

## Install

```bash
pip install numpy pillow pandas tqdm psutil zstandard
```

## Usage

Basic processing:

```bash
python main.py \
  --input_dir ./exploration_output/ \
  --output_dir ./processed_output/
```

Memory-aware processing:

```bash
python main.py \
  --input_dir ./exploration_output/ \
  --output_dir ./processed_output/ \
  --memory_limit 16 \
  --batch_size 3 \
  --verbose
```

Low-memory mode:

```bash
python main.py \
  --input_dir ./exploration_output/ \
  --output_dir ./processed_output/ \
  --memory_limit 8 \
  --batch_size 1 \
  --no_screenshots
```

Lightweight preview:

```bash
python main.py \
  --input_dir ./exploration_output/ \
  --output_dir ./preview_output/ \
  --lightweight
```

Process one app:

```bash
python main.py \
  --input_dir ./exploration_output/ \
  --output_dir ./processed_output/ \
  --app_package com.example.app
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--input_dir` | Input exploration-output directory | `./exploration_output` |
| `--output_dir` | Output directory | `./exploration_output_uncompress` |
| `--app_package` | App package to process | all apps |
| `--memory_limit` | Memory limit in GB | `16.0` |
| `--batch_size` | Number of trajectories per batch | `3` |
| `--max_trajectories` | Maximum trajectories per app; `0` means unlimited | `0` |
| `--no_screenshots` | Skip screenshot extraction | `False` |
| `--no_html` | Skip HTML report generation | `False` |
| `--no_stats` | Skip statistics generation | `False` |
| `--verbose` | Enable verbose logging | `False` |
| `--lightweight` | Process only a small preview per app | `False` |
| `--aggressive_cleanup` | Run more frequent memory cleanup | `False` |
| `--skip_large_apps` | Skip apps estimated to exceed the memory limit | `False` |

## Output Layout

```text
output_dir/
`-- app_package_name/
    |-- app_info.json
    |-- statistics.json
    |-- trajectories_summary.json
    |-- trajectories/
    |   `-- trajectory_*.json
    `-- screenshots/
        `-- trajectory_id/
            `-- step_*/
                `-- *.png
```

## Troubleshooting

- Out of memory: reduce `--batch_size`, use `--no_screenshots`, or lower `--max_trajectories`.
- Slow processing: increase `--batch_size` if memory allows, or use `--no_screenshots`.
- Incomplete outputs: process one app at a time with `--app_package` and inspect logs with `--verbose`.
