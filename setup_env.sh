#!/bin/bash
# create cuda env
conda create -n DescEdit python==3.10
conda activate DescEdit 

# check cuda version
nvidia-smi
nvcc -V 
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-deps 
pip install xformers==0.0.28 -i https://download.pytorch.org/whl/cu118
pip install datasets omegaconf matplotlib scikit-learn
# remove cuda env
# conda remove -n DescEdit --all  
