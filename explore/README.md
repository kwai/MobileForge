# MobileForge Explore

This directory contains the target-app exploration and MobileGym-Curriculum task-generation code used by MobileForge.

## Setup

```bash
conda create -n mobileforge-explore python=3.12 -y
conda activate mobileforge-explore
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then fill in your model endpoint and API key. Exploration expects the local embedding and retrieval services to be available:

```bash
python -m utils.embedding_pipeline
python -m utils.retrieval
```

Before running exploration, prepare the GUI-explorer knowledge base and place target APK files under `apks/`, or pull them from a connected device.

## Target-App Exploration

Interactive multi-device exploration:

```bash
python interactive_parallel_exploration.py
```

Command-line parallel exploration:

```bash
python -m parallel_exploration.main --config parallel_exploration/parallel_config.yaml.example
```

Single-device exploration:

```bash
python exploration_and_mining.py \
  -device_serial emulator-5554 \
  -package_name com.android.settings \
  -output_dir exploration_output \
  -max_branching_factor 10 \
  -max_exploration_steps 30 \
  -max_exploration_depth 5
```

Each app output includes `app_info.json`, step screenshots, UI elements, actions, summaries, and compressed trajectory files.

## Curriculum Generation

The paper experiments use `curriculum_generator_refactored_add_aw_few-shot/`. It reads exploration traces, adds AndroidWorld few-shot examples, and generates rollout-ready curriculum tasks.

```bash
python curriculum_generator_refactored_add_aw_few-shot/main.py \
  --input_dir /path/to/exploration_output_final \
  --output_dir generated_tasks
```

Main outputs:

- `generated_tasks.csv`: per-app generated tasks.
- `all_generated_tasks.csv`: merged tasks across apps.
- `statistics.json` and `master_summary.md`: task-generation statistics and summaries.

The generated tasks used in the paper are released at [🤗 `lgy0404/mobileforge-generated-tasks`](https://huggingface.co/datasets/lgy0404/mobileforge-generated-tasks).

## Released Exploration Data

The paper exploration trajectories are released in two splits:

- `raw`: raw exploration outputs.
- `final`: parsed and visualized exploration outputs.

Released dataset: [🤗 `lgy0404/mobileforge-exploration-trajectories`](https://huggingface.co/datasets/lgy0404/mobileforge-exploration-trajectories).
