# Parallel App Exploration

This module extends single-device exploration to multiple Android devices or emulators. It assigns apps to available devices, runs exploration jobs in parallel, and keeps device failures isolated from the full batch.

## Features

- Multi-device exploration with physical devices or cloned AVD instances.
- Automatic task scheduling and load balancing.
- Per-app isolation to reduce cross-app interference.
- Device health checks and retry logic.
- Batch-mode exploration over a package list.

## Prerequisites

Set up the MobileForge Explore environment first:

```bash
conda activate mobileforge-explore
python -m utils.embedding_pipeline &
python -m utils.retrieval &
```

Prepare either physical Android devices with ADB enabled, or an AndroidWorld-compatible source AVD.

## Quick Start

Run from the `explore/` directory:

```bash
# Single-device exploration
python run_parallel_exploration.py -package_name com.android.camera2

# Batch exploration on two physical devices
python run_parallel_exploration.py -batch_mode -num_devices 2

# Batch exploration on three emulator instances
python run_parallel_exploration.py -batch_mode -num_devices 3 -use_emulator \
  -emulator_exe /path/to/emulator \
  -source_avd_name AndroidWorldAvd \
  -source_avd_home /path/to/avd/home \
  -android_sdk_path /path/to/android-sdk
```

Run from this directory:

```bash
python main.py -package_name com.android.camera2
python main.py -batch_mode -num_devices 2
```

Convenience script:

```bash
./run_parallel_exploration.sh --help
./run_parallel_exploration.sh -b -n 3 -e \
  --emulator-exe /path/to/emulator \
  --avd-name AndroidWorldAvd \
  --avd-home /path/to/avd
```

## Common Arguments

Exploration:

- `-package_name`: package name for single-app exploration.
- `-device_serial`: specific device serial.
- `-output_dir`: output directory; defaults to `./exploration_output_parallel`.
- `-max_branching_factor`: maximum goals per node.
- `-max_exploration_steps`: maximum steps per goal.
- `-max_exploration_depth`: maximum exploration depth.

Parallel execution:

- `-num_devices`: number of devices.
- `-use_emulator`: use cloned emulator instances.
- `-emulator_exe`: emulator executable path.
- `-source_avd_name`: source AVD name.
- `-source_avd_home`: source AVD directory.
- `-android_sdk_path`: Android SDK path.

Batch mode:

- `-batch_mode`: enable package-list exploration.
- `-app_list_file`: package-list file.
- `-skip_knowledge_extraction`: skip post-exploration knowledge extraction.

Example `app_packages.txt`:

```text
com.android.camera2
com.android.settings
com.google.android.deskclock
com.arduia.expense
net.gsantner.markor
```

## Environment Variables

Optional `.env` entries:

```bash
PARALLEL_MAX_RETRY=2
PARALLEL_DEVICE_TIMEOUT=120
PARALLEL_LOG_LEVEL=INFO
```

## Troubleshooting

Check AVDs:

```bash
emulator -list-avds
```

Check occupied ports:

```bash
lsof -i :5554
```

Check ADB devices:

```bash
adb devices
adb kill-server && adb start-server
adb -s emulator-5554 shell echo alive
```

If memory is limited, reduce `-num_devices` or lower exploration depth and branching settings.
