# MML - SurgAdapt

This repository contains the codebase for **MML - SurgAdapt**, an adaptation of the CLIP for surgery. The project is designed for multi-task surgical computer vision and supports easy setup, training, and inference.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Running Training](#running-training)
3. [Running Inference](#running-inference)
4. [Information regarding Configs](#information-regarding-configs)
5. [Pretrained Weights](#pretrained-weights)

---

## Environment Setup

Follow these steps to set up the environment:

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/CAMMA-public/MMA-SurgAdapt.git
   cd MMA-SurgAdapt

2. **Create a Python Virtual Environment**
   ```bash
   conda create -n env python=3.12
   conda activate env

3. **Install Dependencies**
   ```bash
   conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt

## Running Training

For training the model, use the config file `configs/surgadapt+cholec.yaml` to set the configuration for training and run:
```bash
python train.py
```

## Running Inference

For testing the model, use the config file `configs/surgadapt+cholec.yaml` to set the configuration for testing and run:
```bash
python test.py
```

## Information regarding Configs

Coming soon ...

## Pretrained Weights

Coming soon ...
