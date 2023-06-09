# !/bin/bash
conda create -n donut_official python=3.11
conda activate donut_official

pip install donut-python
pip install gradio
pip install timm==0.6.13
