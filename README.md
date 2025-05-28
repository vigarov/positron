# Automatic Processing of DSLR-Scanned Film Negatives

This repository contains the code surrounding the Computational Photography course project aiming to automate the processingof DSLR-scanned film negatives. 


## Project Structure

```
.
├── data/                   # Data storage and organization
│   ├── raw/               # Raw camera files (.RAF, .HIF)
│   ├── prepro/            # Pre-processed images  
│   ├── datasets/          # Training/validation datasets
│   ├── checkpoints/       # Model checkpoints
│   └── reproduction/      # Reproduction assets
├── src/                   # Source code
│   ├── nn/               # Machine Learning approach
│   │   ├── models/       # Neural network architectures
│   │   ├── utils/        # Training utilities and dataset loading
│   │   ├── eval/         # Evaluation and testing tools
│   │   └── data/         # Data handling
│   ├── scripts/          # Processing and utility scripts. Includes training scripts.
│   │   ├── sigmoid/      # Statistical sigmoid correction
│   │   ├── kp_calc/      # Tool to collect the color dataset by
│   │   └── train/        # Training scripts
│   ├── processing/       # Statistical image processing. Contains the Sigmoid correct code
│   ├── analysis/         # Analysis and visualization tools
│   ├── utils/           # General utilities
└── requirements.txt
```

## Machine Learning Approach (`src/nn/`)

The neural network pipeline provides end-to-end learning for film negative correction:

- The model architectures are located in src/nn/models (`one_hidden`, `enhanced_comb`, `combhidden`) 
- Full training pipeline with validation and testing through the `main_trainer.py` script (see `src/scripts/training` for an example invocation)
  - Training in both RGB and LAB color spaces has been tested (can be switched with a flag) 

## Statistical Processing (`src/scripts/sigmoid/` and `src/processing/`)

- White Balance methods can be selected using the `WhiteBalance` class (`white_balance.py`).
- Hyper-parameter optimization (namely generating a PDF with various combinations of hyperparameters, which can be selected through a configuration file) through the `src/scripts/sigmoid/correct_many.py`

## Data Collection Tool (`src/scripts/kp_calc/`)

Interactive GUI application for data collection and annotation

- Using image mappings, draw matching rectangles to select pixel regions across image sets
- Adapts based on image transforms
- Used to generate CSV datasets for training the NN models (see `--multiple` and `--data-processing` flags)

Usage:
```bash
python src/scripts/kp_calc/app.py
```

## Recommended setup

### Prerequisites
- Python 3.9+ (recommended 3.12+, tested with 3.10+)
- PyTorch if NN training(install according to your system: https://pytorch.org/get-started/locally/)
- exiftool (required for RAW file metadata extraction)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/vigarov/positron
cd positron
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data directory structure (see `data/README.md` for details)

### Quick Start Workflow

1. **Data Collection**: Use the interactive tool to collect training data
```bash
python src/scripts/kp_calc/kp_calc.py -h
```

2. **Statistical Processing**: Generate a PDF with applying the sigmoid correction method on all the images from the dataset
```bash
python src/scripts/sigmoid/correct_many.py -h
```

3. **ML Training**: Train a neural network model

Simply execute one of the trainers in `src/scripts/train/` or run your own train by invoking the train loop:
```bash
python src/nn/main_trainer.py -h
```
