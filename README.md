# File structure
Here are some important folder and file locations:
- FSS_implantation/
  - dataset/
    - multi_thread_flexible_electrode/
    - create_synthetic_image.py
  - FSS_Hooking/
    - changed_torchvision/
    - main.py
    - init.pt
  - requirements.txt
  - README.md

# Dataset
Download dataset from https://drive.google.com/file/d/1oTtLOOb0mAUO_yayhHi7k9dKdbNCp0-w/view?usp=sharing

# Enviroment
**We changed torchvision library, so we need to replace some files.**
1. pip install -r requirements.txt
2. Copy and overwrite the files in "changed_torchvision/" to "torchvision/".

# Training and evaluation
0. Download "init.pt" from https://drive.google.com/drive/folders/1E15Svoja5mSd16cogmsnSYWkyx4vD8-Q?usp=sharing
1. Set parameters in "create_synthetic_image.py" and run it. In our experiments, we generate images until we have 100 training images and set training epochs to 300.
2. Set parameters in "main.py" and run it.

# Model weights of proposed methods used in our experiments.
5 shot: https://drive.google.com/drive/folders/1E15Svoja5mSd16cogmsnSYWkyx4vD8-Q?usp=sharing
