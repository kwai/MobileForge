#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# shellcheck disable=SC1091
source activate base

conda create -n MobileForge python=3.11.0 -y
echo "Created conda environment MobileForge with Python 3.11.0."

conda activate MobileForge
pip install -r requirements.txt
echo "Installed packages in MobileForge from requirements.txt."


### UITARS、UITARS_1_5 ###
name="android_world"
python_version="3.11.8"
conda create -n $name python=$python_version -y
echo "Created conda environment ${name} with Python ${python_version}."
conda activate $name
cd ./framework/android_env
python setup.py install
echo "Successfully run setup.py for AndroidEnv"
cd ../../
pip install -r ./requirements/${name}.txt
pip install protobuf==5.29.0
cd ./framework/models/AndroidWorld
python setup.py install
echo "Successfully run setup.py for AndroidWorld"
cd ../../../
echo "Installed packages in ${name}."
conda deactivate


# echo "Script completed successfully."
