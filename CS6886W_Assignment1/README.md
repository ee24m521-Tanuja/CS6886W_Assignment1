# CS6886W Assignment 1: VGG6 on CIFAR-10

Repository for **CS6886W - Systems Engineering for Deep Learning, Assignment 1**.

GitHub repository: https://github.com/ee24m521-Tanuja/CS6886W_Assignment1.git

## Objective

Train a VGG6-style CNN on CIFAR-10 and study the effect of activation function, optimizer, batch size, number of epochs, and learning rate. Experiments are tracked with Weights & Biases (W&B).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── model.py
├── data.py
├── engine.py
├── optimizers.py
├── augmentations.py
├── activations/
│   ├── __init__.py
│   ├── relu.py
│   ├── sigmoid.py
│   ├── tanh.py
│   ├── silu.py
│   └── gelu.py
├── checkpoints/
│   ├── README.md
│   └── best_model.pth
├── plots/
│   ├── 01_parallel_coordinates.png
│   ├── 02_validation_accuracy_scatter.png
│   ├── 03_training_accuracy.png
│   ├── 04_training_loss.png
│   ├── 05_validation_accuracy.png
│   └── 06_validation_loss.png
└── results/
    ├── results_summary.csv
    └── wandb_runs.csv
```

## Model

The VGG6-style feature configuration used is:

```python
[64, 64, 'M', 128, 128, 'M']
```

Each convolution block uses a 3x3 convolution and Batch Normalization followed by the selected activation. `M` denotes 2x2 max pooling. Adaptive average pooling produces a 128-dimensional feature vector which is classified using a `Linear(128, 10)` layer.

## CIFAR-10 preprocessing and augmentation

Training transforms:

1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()`
3. `ToTensor()`
4. `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`
5. `Cutout(n_holes=1, length=16)`

Validation/test transforms:

1. `ToTensor()`
2. The same CIFAR-10 normalization

## Environment used

- Python: **3.13.15** (GCC 13.3.0)
- PyTorch: **2.11.0+cu128**
- Torchvision: **0.26.0+cu128**
- NumPy: **2.1.3**
- Weights & Biases: **0.28.1**
- CUDA: **12.8**
- Seed: **42**

The experiments were executed in CUDA-enabled Google Colab and Kaggle environments.

> `requirements.txt` uses the package versions corresponding to this environment.

## Setup

```bash
git clone https://github.com/ee24m521-Tanuja/CS6886W_Assignment1.git
cd CS6886W_Assignment1

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Login to W&B if experiment logging is required:

```bash
wandb login
```

## Reproduce the reported best model

The best measured configuration is:

| Parameter | Value |
|---|---|
| Model | VGG6 |
| Dataset | CIFAR-10 |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 100 |
| Seed | 42 |
| Best epoch | 97 |
| Best validation accuracy | **88.12%** |
| Final validation accuracy | 86.61% |

Run:

```bash
python3 train.py --activation relu --optimizer adam --epochs 100 --batch-size 64 --lr 0.01 --seed 42 --workers 2 --output results/best_model --wandb
```

The experiment recorded a best validation accuracy of **88.12% at epoch 97**.

## Baseline

Baseline configuration: ReLU + Adam, learning rate 0.001, batch size 64, 100 epochs, seed 42.

- Best validation accuracy: **86.71%**
- Best epoch: **84**
- Final validation accuracy: **85.75%**

Run:

```bash
python3 train.py --activation relu --optimizer adam --epochs 100 --batch-size 64 --lr 0.001 --seed 42 --workers 2 --output results/baseline --wandb
```

## Activation-function comparison

| Activation | Best validation accuracy |
|---|---:|
| GELU | **87.58%** |
| SiLU | 87.20% |
| ReLU | 86.87% |
| Tanh | 81.56% |
| Sigmoid | 77.50% |

GELU achieved the highest accuracy among the controlled activation experiments, followed closely by SiLU and ReLU. Tanh and especially Sigmoid produced lower validation accuracy.

## Optimizer comparison

| Optimizer | Learning rate | Best validation accuracy |
|---|---:|---:|
| RMSprop | 0.001 | **87.96%** |
| Nesterov-SGD | 0.01 | 86.91% |
| SGD | 0.01 | 86.77% |
| Adam | 0.001 | 86.71% |
| NAdam | 0.001 | 86.66% |
| Adagrad | 0.01 | 80.49% |

RMSprop was strongest in the optimizer comparison. SGD and Nesterov-SGD reached competitive accuracy but their best values occurred late in training. Adagrad underperformed the other tested optimizers.

## Batch-size comparison

| Batch size | Best validation accuracy |
|---:|---:|
| 32 | **87.66%** |
| 64 | 86.71% |
| 128 | 85.72% |

Batch size 32 achieved the highest result in this controlled comparison, while accuracy decreased as batch size increased to 128.

## Epoch comparison

| Epochs | Best validation accuracy |
|---:|---:|
| 50 | 84.96% |
| 100 | 86.71% |
| 150 | **87.76%** |

Longer training improved the best observed validation accuracy, and the 150-epoch run continued improving to epoch 145.

## Learning-rate comparison

| Learning rate | Best validation accuracy |
|---:|---:|
| 0.0001 | 80.88% |
| 0.001 | 86.71% |
| 0.01 | **88.12%** |

The learning rate had a strong effect on convergence. `0.0001` learned substantially more slowly, while `0.01` produced the overall highest measured validation accuracy.

## W&B plots

The `plots/` directory contains the required W&B visualizations:

1. Parallel-coordinates plot
2. Validation accuracy vs. step scatter plot
3. Training accuracy vs. step
4. Training loss vs. step
5. Validation accuracy vs. step
6. Validation loss vs. step

The consolidated W&B run table is included as `results/wandb_runs.csv`, and `results/results_summary.csv` contains the completed full experiments sorted for analysis.

## Reproducibility

`train.py` accepts `--seed`; all reported controlled experiments use seed **42**. The training code seeds Python, NumPy, CPU PyTorch, and CUDA PyTorch where available.

Training and evaluation are separated in `engine.py`, model construction is in `model.py`, data/transforms are in `data.py` and `augmentations.py`, optimizer construction is in `optimizers.py`, and each requested activation function has its own module in `activations/`.

