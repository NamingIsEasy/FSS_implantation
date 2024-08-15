# File structure
Here are some important folder and file locations:
- FSS_implantation/
  - dataset/
    - multi_thread_flexible_electrode/
    - create_synthetic_image.py
  - FSS_Hooking/
    - changed_torchvision/
    - main.py
  - requirements.txt
  - README.md

# Dataset
Download dataset from https://drive.google.com/file/d/1oTtLOOb0mAUO_yayhHi7k9dKdbNCp0-w/view?usp=sharing

# Enviroment
**We changed torchvision library, so we need to replace some files.**
1. pip install -r requirements.txt
2. Copy and overwrite the files in "changed_torchvision/" to "torchvision/".

# 

# Training and evaluation
