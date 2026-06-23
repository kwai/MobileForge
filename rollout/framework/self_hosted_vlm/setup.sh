#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# shellcheck disable=SC1091
source activate base
conda create -n swift python=3.10 -y
echo "Created conda environment swift with Python 3.10."

conda activate swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# Install decord for internlm-xcomposer2_5-7b-chat
pip install decord
