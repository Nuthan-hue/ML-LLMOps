# Model Finetuning Project Structure

## Overview
This document outlines the recommended folder structure for implementing model finetuning in this ML-LLMOps project.

## Complete Directory Structure

```
ML-LLMOps/
├── data/                           # Data storage
│   ├── raw/                        # Original, immutable datasets
│   ├── processed/                  # Cleaned and preprocessed data
│   └── finetuning/                 # Finetuning-ready datasets
│       ├── train.jsonl             # Training data
│       ├── val.jsonl               # Validation data
│       └── test.jsonl              # Test data
│
├── src/                            # Source code
│   ├── data/                       # Data pipeline modules
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Fetch/download raw data
│   │   ├── data_preprocessing.py   # Clean and transform data
│   │   └── prepare_finetuning_data.py  # Format data for finetuning
│   │
│   ├── models/                     # Model-related code
│   │   ├── __init__.py
│   │   ├── base_model.py           # Load pretrained models
│   │   ├── finetune.py             # Finetuning logic
│   │   ├── inference.py            # Model inference
│   │   └── model_registry.py       # Model version management
│   │
│   ├── training/                   # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop
│   │   ├── callbacks.py            # Training callbacks
│   │   ├── hyperparameters.py      # Hyperparameter configs
│   │   └── optimizers.py           # Custom optimizers
│   │
│   ├── evaluation/                 # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py              # Custom metrics
│   │   ├── evaluate.py             # Evaluation pipeline
│   │   └── compare_models.py       # Compare base vs finetuned
│   │
│   └── utils/                      # Helper functions
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── logging.py              # Logging utilities
│       └── mlflow_utils.py         # MLflow helpers
│
├── configs/                        # Configuration files
│   ├── base_config.yaml            # Base configuration
│   ├── data_config.yaml            # Data processing config
│   └── finetuning/                 # Finetuning configurations
│       ├── lora_config.yaml        # LoRA (Low-Rank Adaptation)
│       ├── full_finetune.yaml      # Full model finetuning
│       ├── qlora_config.yaml       # Quantized LoRA
│       └── adapter_config.yaml     # Adapter-based finetuning
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Explore raw data
│   ├── 02_baseline_model.ipynb     # Test pretrained model
│   ├── 03_finetuning_experiments.ipynb  # Finetuning experiments
│   └── 04_model_comparison.ipynb   # Compare model versions
│
├── models/                         # Saved models
│   ├── pretrained/                 # Downloaded pretrained models
│   ├── checkpoints/                # Training checkpoints
│   │   ├── epoch_1/
│   │   ├── epoch_2/
│   │   └── best_model/
│   └── final/                      # Production-ready models
│       └── model_v1/
│
├── experiments/                    # MLflow experiment artifacts
│   └── mlruns/                     # MLflow tracking data
│
├── scripts/                        # Executable scripts
│   ├── download_pretrained.sh      # Download base models
│   ├── prepare_data.sh             # Data preparation pipeline
│   ├── train.sh                    # Training script
│   ├── evaluate.sh                 # Evaluation script
│   └── deploy.sh                   # Deployment script
│
├── tests/                          # Unit tests
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_training.py
│
├── docs/                           # Documentation
│   ├── setup.md                    # Setup instructions
│   ├── finetuning_guide.md         # Finetuning guide
│   └── api_reference.md            # API documentation
│
├── .dvc/                           # DVC configuration
├── .git/                           # Git repository
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Hyperparameters & configs
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── .gitignore                      # Git ignore rules
├── .dvcignore                      # DVC ignore rules
└── README.md                       # Project overview
```

## Directory Descriptions

### `/data`
**Purpose**: Store all datasets at different processing stages
- `raw/`: Original datasets (never modify)
- `processed/`: Cleaned data ready for feature engineering
- `finetuning/`: Train/val/test splits in model-specific format

### `/src`
**Purpose**: All source code organized by functionality

#### `/src/data`
Data pipeline components:
- Ingestion: Download/fetch data from sources
- Preprocessing: Clean, normalize, handle missing values
- Finetuning prep: Format data for specific model architectures

#### `/src/models`
Model-related code:
- Load pretrained models (HuggingFace, PyTorch Hub, etc.)
- Finetuning implementation (LoRA, QLoRA, full finetuning)
- Inference pipeline for production
- Model versioning and registry

#### `/src/training`
Training infrastructure:
- Training loops with validation
- Callbacks (early stopping, checkpointing, logging)
- Hyperparameter management
- Custom optimizers and schedulers

#### `/src/evaluation`
Model evaluation:
- Metrics calculation (accuracy, F1, perplexity, etc.)
- Comparison between base and finetuned models
- Generate evaluation reports

#### `/src/utils`
Helper utilities:
- Configuration loading
- Logging setup
- MLflow integration helpers

### `/configs`
**Purpose**: YAML/JSON configuration files
- Separate configs for different finetuning strategies
- Environment-specific configurations
- Hyperparameter sets for experiments

### `/notebooks`
**Purpose**: Interactive exploration and experimentation
- Data analysis and visualization
- Quick prototyping
- Results analysis

### `/models`
**Purpose**: Store model artifacts
- `pretrained/`: Downloaded base models
- `checkpoints/`: Training checkpoints for resuming
- `final/`: Production-ready models

### `/experiments`
**Purpose**: MLflow experiment tracking
- Stores runs, metrics, parameters
- Model artifacts logged by MLflow

### `/scripts`
**Purpose**: Shell scripts for common operations
- Automate training pipelines
- Data download and preparation
- Model evaluation and deployment

## DVC Pipeline Integration

The folder structure integrates with DVC for reproducible pipelines:

```yaml
# dvc.yaml example
stages:
  data_ingestion:
    cmd: python src/data/data_ingestion.py
    deps:
      - src/data/data_ingestion.py
    outs:
      - data/raw

  data_preprocessing:
    cmd: python src/data/data_preprocessing.py
    deps:
      - data/raw
      - src/data/data_preprocessing.py
    outs:
      - data/processed

  prepare_finetuning_data:
    cmd: python src/data/prepare_finetuning_data.py
    deps:
      - data/processed
      - configs/data_config.yaml
    outs:
      - data/finetuning
    params:
      - data_config.train_test_split
      - data_config.validation_split

  finetune_model:
    cmd: python src/models/finetune.py
    deps:
      - data/finetuning
      - configs/finetuning/lora_config.yaml
      - src/models/finetune.py
    outs:
      - models/checkpoints
    params:
      - finetuning.learning_rate
      - finetuning.num_epochs
      - finetuning.batch_size

  evaluate_model:
    cmd: python src/evaluation/evaluate.py
    deps:
      - models/checkpoints/best_model
      - data/finetuning/test.jsonl
      - src/evaluation/evaluate.py
    metrics:
      - metrics.json
```

## Configuration Management with params.yaml

```yaml
# params.yaml example
data_config:
  train_test_split: 0.8
  validation_split: 0.1
  max_samples: 10000

finetuning:
  model_name: "gpt2"  # or "bert-base-uncased", "resnet50", etc.
  method: "lora"  # "lora", "qlora", "full", "adapter"
  learning_rate: 2e-5
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  warmup_steps: 100

  # LoRA specific
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05

  # Training
  max_length: 512
  save_steps: 500
  eval_steps: 500
  logging_steps: 100

evaluation:
  metrics: ["accuracy", "f1", "precision", "recall"]
  generate_report: true
```

## MLflow Integration

This structure supports MLflow experiment tracking:

```python
# Example in src/models/finetune.py
import mlflow
import mlflow.pytorch

# Start MLflow run
with mlflow.start_run(experiment_name="model-finetuning"):
    # Log parameters
    mlflow.log_params(config)

    # Log metrics during training
    mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("val_accuracy", accuracy, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "finetuned_model")

    # Log artifacts
    mlflow.log_artifact("configs/finetuning/lora_config.yaml")
```

## Common Finetuning Methods

### 1. Full Finetuning
- Train all model parameters
- Requires significant compute
- Best performance but highest resource cost

### 2. LoRA (Low-Rank Adaptation)
- Train small adapter layers
- Freeze base model weights
- Efficient, good performance

### 3. QLoRA (Quantized LoRA)
- Quantize base model to 4-bit
- Add LoRA adapters
- Most memory efficient

### 4. Adapter Tuning
- Add small adapter modules between layers
- Freeze base model
- Modular and efficient

## Getting Started

1. **Setup environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Initialize DVC and MLflow**
   ```bash
   dvc init
   mlflow ui  # Start MLflow UI
   ```

3. **Prepare data**
   ```bash
   dvc repro data_ingestion
   dvc repro data_preprocessing
   dvc repro prepare_finetuning_data
   ```

4. **Finetune model**
   ```bash
   dvc repro finetune_model
   ```

5. **Evaluate**
   ```bash
   dvc repro evaluate_model
   dvc metrics show
   ```

## Best Practices

1. **Version Control**
   - Git for code and configs
   - DVC for data and models
   - MLflow for experiments

2. **Reproducibility**
   - Pin dependencies in requirements.txt
   - Use params.yaml for all hyperparameters
   - Document DVC pipeline in dvc.yaml

3. **Experiment Tracking**
   - Log all hyperparameters
   - Track metrics at each step
   - Save model checkpoints regularly

4. **Data Management**
   - Keep raw data immutable
   - Version datasets with DVC
   - Document data transformations

5. **Model Management**
   - Save checkpoints during training
   - Version final models
   - Document model performance

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)