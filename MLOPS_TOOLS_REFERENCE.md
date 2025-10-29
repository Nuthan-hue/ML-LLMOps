# MLOps Tools Quick Reference Guide

> Quick reference for all MLOps tools, commands, and configurations

## Table of Contents

1. [DVC (Data Version Control)](#dvc-data-version-control)
2. [MLflow](#mlflow)
3. [DagsHub](#dagshub)
4. [Docker & Docker Compose](#docker--docker-compose)
5. [Hugging Face](#hugging-face)
6. [Ollama](#ollama)
7. [Git Best Practices for ML](#git-best-practices-for-ml)
8. [Python Environment Management](#python-environment-management)

---

## DVC (Data Version Control)

### Installation
```bash
pip install dvc
pip install dvc[s3]  # For S3
pip install dvc[gs]  # For Google Cloud
pip install dvc[azure]  # For Azure
```

### Initialize DVC
```bash
# In your git repo
dvc init

# Commit DVC configuration
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Basic Commands

#### Track Data/Models
```bash
# Track a file or directory
dvc add data/raw/dataset.csv
dvc add models/model.pkl
dvc add data/processed/

# This creates .dvc files - commit them to git
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add dataset"
```

#### Remote Storage
```bash
# Add remote storage
dvc remote add -d storage s3://mybucket/dvcstore
dvc remote add -d storage gs://mybucket/dvcstore
dvc remote add -d storage /tmp/dvcstore  # Local

# For DagsHub
dvc remote add origin https://dagshub.com/USER/REPO.dvc

# Configure credentials (S3 example)
dvc remote modify storage access_key_id 'mykey'
dvc remote modify storage secret_access_key 'mysecret'

# List remotes
dvc remote list

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

#### Pipeline Management

**dvc.yaml**:
```yaml
stages:
  data_ingestion:
    cmd: python src/data/ingestion.py
    deps:
      - src/data/ingestion.py
    outs:
      - data/raw/dataset.csv

  preprocessing:
    cmd: python src/data/preprocessing.py
    deps:
      - data/raw/dataset.csv
      - src/data/preprocessing.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - preprocessing.train_ratio
      - preprocessing.test_ratio

  train:
    cmd: python src/models/train.py
    deps:
      - data/processed/train.csv
      - src/models/train.py
    outs:
      - models/model.pkl
    params:
      - train.learning_rate
      - train.n_estimators
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/test.csv
      - src/models/evaluate.py
    metrics:
      - reports/metrics.json:
          cache: false
```

**params.yaml**:
```yaml
preprocessing:
  train_ratio: 0.8
  test_ratio: 0.2

train:
  learning_rate: 0.01
  n_estimators: 100
  max_depth: 10
```

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train

# Force re-run (ignore cache)
dvc repro -f

# Show pipeline DAG
dvc dag

# Check status
dvc status

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff HEAD~1 HEAD
dvc params diff
```

#### Experiments
```bash
# Run experiment with different parameters
dvc exp run -S train.learning_rate=0.001

# List experiments
dvc exp list

# Show experiment results
dvc exp show

# Compare experiments
dvc exp diff

# Apply best experiment
dvc exp apply exp-abc123

# Remove experiments
dvc exp remove exp-abc123
```

#### Useful Commands
```bash
# Check out data for specific git commit
git checkout <commit-hash>
dvc checkout

# Garbage collection (cleanup unused cache)
dvc gc

# Check DVC config
dvc config -l

# Pull specific file
dvc pull data/raw/dataset.csv.dvc
```

---

## MLflow

### Installation
```bash
pip install mlflow
pip install mlflow[extras]  # With additional dependencies
```

### Start MLflow UI
```bash
# Local
mlflow ui

# Specify port and host
mlflow ui --host 0.0.0.0 --port 5000

# With specific backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Python API

#### Basic Logging
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
# Or for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/USER/REPO.mlflow")

# Set experiment
mlflow.set_experiment("my-experiment")

# Start run
with mlflow.start_run(run_name="experiment-1"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05, step=1)  # With step for time series

    # Log multiple metrics
    mlflow.log_metrics({"precision": 0.9, "recall": 0.92})

    # Log artifacts (files)
    mlflow.log_artifact("plot.png")
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifacts("outputs/")  # Entire directory

    # Set tags
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tags({"version": "v1", "stage": "development"})
```

#### Log Models
```python
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow

# Scikit-learn
mlflow.sklearn.log_model(model, "model")

# PyTorch
mlflow.pytorch.log_model(model, "model")

# TensorFlow
mlflow.tensorflow.log_model(model, "model")

# Custom Python function
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=custom_model,
    artifacts={"tokenizer": "tokenizer.pkl"}
)
```

#### Load Models
```python
import mlflow

# Load from run
model = mlflow.sklearn.load_model("runs:/<run-id>/model")

# Load from model registry
model = mlflow.pyfunc.load_model("models:/my-model/Production")
model = mlflow.pyfunc.load_model("models:/my-model/1")  # Specific version

# Predict
predictions = model.predict(X_test)
```

#### Model Registry
```python
# Register model
mlflow.register_model("runs:/<run-id>/model", "my-model")

# Or during logging
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="my-model"
)

# Transition model stage
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Production"
)
# Stages: None, Staging, Production, Archived
```

#### Autologging
```python
import mlflow

# Sklearn
mlflow.sklearn.autolog()

# PyTorch
mlflow.pytorch.autolog()

# TensorFlow
mlflow.tensorflow.autolog()

# XGBoost
mlflow.xgboost.autolog()

# LightGBM
mlflow.lightgbm.autolog()

# Then just train your model - everything logged automatically!
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

#### Search Runs
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("my-experiment")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics['accuracy']}")
```

### CLI Commands
```bash
# List experiments
mlflow experiments list

# Create experiment
mlflow experiments create -n "new-experiment"

# Delete experiment
mlflow experiments delete -x <experiment-id>

# List runs
mlflow runs list --experiment-id 0

# Serve model
mlflow models serve -m "models:/my-model/Production" -p 5001

# Deploy model (various platforms)
mlflow deployments help
```

---

## DagsHub

### Setup
```python
import dagshub

# Initialize
dagshub.init(
    repo_owner='username',
    repo_name='repo-name',
    mlflow=True  # Enable MLflow tracking
)

# Now use MLflow as normal - it will sync to DagsHub
import mlflow
mlflow.log_param("test", 123)
```

### Configure Git Remote
```bash
# Add DagsHub as remote
git remote add origin https://dagshub.com/username/repo-name.git

# Or with credentials
git remote add origin https://username:token@dagshub.com/username/repo-name.git
```

### DVC with DagsHub
```bash
# Add DagsHub as DVC remote
dvc remote add origin https://dagshub.com/username/repo-name.dvc

# Configure credentials
dvc remote modify origin --local auth basic
dvc remote modify origin --local user username
dvc remote modify origin --local password <your-token>

# Push data
dvc push -r origin
```

### Get DagsHub Token
1. Go to https://dagshub.com/user/settings/tokens
2. Create new token
3. Use in git remote or DVC config

---

## Docker & Docker Compose

### Dockerfile for ML Projects

**For Traditional ML:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "src/api/app.py"]
```

**For Deep Learning (with CUDA):**
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/api/app.py"]
```

**For Hugging Face Models:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN pip install transformers torch peft accelerate

COPY . .

EXPOSE 8000

CMD ["python", "src/inference/api.py"]
```

### Docker Commands
```bash
# Build image
docker build -t my-ml-app:latest .

# Build with build args
docker build --build-arg MODEL_NAME=bert-base -t my-ml-app .

# Run container
docker run -p 8000:8000 my-ml-app

# Run with volume mount (for models)
docker run -v $(pwd)/models:/app/models -p 8000:8000 my-ml-app

# Run with GPU
docker run --gpus all -p 8000:8000 my-ml-app

# Run in background
docker run -d -p 8000:8000 my-ml-app

# View logs
docker logs <container-id>
docker logs -f <container-id>  # Follow

# Stop container
docker stop <container-id>

# Remove container
docker rm <container-id>

# Remove image
docker rmi my-ml-app

# List images
docker images

# List containers
docker ps  # Running
docker ps -a  # All

# Execute command in container
docker exec -it <container-id> bash
```

### docker-compose.yml

**Complete ML Stack:**
```yaml
version: '3.8'

services:
  # ML API
  ml-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - ENV=production
    depends_on:
      - mlflow
      - postgres

  # MLflow Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow
    environment:
      - BACKEND_STORE_URI=postgresql://user:password@postgres:5432/mlflow
      - ARTIFACT_ROOT=/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://user:password@postgres:5432/mlflow
      --default-artifact-root /mlflow
    depends_on:
      - postgres

  # Ollama (for LLMs)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # PostgreSQL (for MLflow backend)
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Streamlit Dashboard
  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - ml-api

volumes:
  ollama_data:
  postgres_data:
```

### Docker Compose Commands
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs
docker-compose logs ml-api  # Specific service
docker-compose logs -f  # Follow

# Execute command in service
docker-compose exec ml-api bash

# Scale service
docker-compose up -d --scale ml-api=3

# List services
docker-compose ps
```

---

## Hugging Face

### Installation
```bash
pip install transformers
pip install transformers[torch]  # With PyTorch
pip install datasets
pip install accelerate
pip install peft  # For LoRA
pip install bitsandbytes  # For QLoRA
```

### CLI Login
```bash
# Login to Hugging Face
huggingface-cli login

# Or set token as environment variable
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### Load Models
```python
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# For specific tasks
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# For text generation
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto",  # Automatic device placement
    torch_dtype="auto"  # Automatic dtype
)

# Load from local path
model = AutoModel.from_pretrained("./my-model")
```

### Use Pipeline (Quick Inference)
```python
from transformers import pipeline

# Classification
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# Question answering
qa = pipeline("question-answering")
result = qa(question="What is ML?", context="Machine learning is...")

# Available tasks:
# - sentiment-analysis
# - text-generation
# - question-answering
# - summarization
# - translation
# - fill-mask
# - zero-shot-classification
```

### Fine-tuning with Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### LoRA Fine-tuning
```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train (only LoRA parameters)
trainer = Trainer(model=model, ...)
trainer.train()

# Save LoRA adapters
model.save_pretrained("./lora-adapters")

# Load LoRA adapters
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "./lora-adapters")
```

### Upload to Hub
```python
# Upload model
model.push_to_hub("username/model-name")
tokenizer.push_to_hub("username/model-name")

# Upload with custom commit message
model.push_to_hub(
    "username/model-name",
    commit_message="Improve accuracy to 95%"
)

# Upload dataset
from datasets import load_dataset
dataset = load_dataset("csv", data_files="data.csv")
dataset.push_to_hub("username/dataset-name")
```

### Download from Hub
```python
# Download specific file
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="username/repo",
    filename="model.safetensors",
    cache_dir="./cache"
)

# Download entire repo
from huggingface_hub import snapshot_download

repo_path = snapshot_download(
    repo_id="username/repo",
    cache_dir="./cache"
)
```

---

## Ollama

### Installation
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from ollama.com

# Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
```

### Basic Commands
```bash
# Pull a model
ollama pull llama2
ollama pull mistral
ollama pull codellama
ollama pull smollm2

# List local models
ollama list

# Run model interactively
ollama run llama2

# Run with prompt
ollama run llama2 "Explain machine learning"

# Remove model
ollama rm llama2

# Show model info
ollama show llama2
```

### Create Custom Model (Modelfile)

**Modelfile**:
```
# Base model
FROM llama2

# System prompt
SYSTEM You are a helpful AI assistant specialized in data science.

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# Template (optional)
TEMPLATE """{{ .System }}
User: {{ .Prompt }}
Assistant: """
```

```bash
# Create model
ollama create my-custom-model -f Modelfile

# Use it
ollama run my-custom-model
```

### Python API
```python
import ollama

# Generate text
response = ollama.generate(
    model='llama2',
    prompt='Explain neural networks'
)
print(response['response'])

# Chat
response = ollama.chat(
    model='llama2',
    messages=[
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Hi! How can I help?'},
        {'role': 'user', 'content': 'Explain ML'}
    ]
)
print(response['message']['content'])

# Stream response
for chunk in ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Write a story'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### REST API
```bash
# Generate
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Explain ML"
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

---

## Git Best Practices for ML

### .gitignore for ML
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# ML specific
*.pkl
*.h5
*.pth
*.onnx
*.pb
models/
!models/.gitkeep

# Data
data/
!data/.gitkeep
*.csv
*.parquet
*.feather

# DVC
/data
/models

# MLflow
mlruns/
mlartifacts/

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/add-new-model

# Make changes and commit
git add src/models/new_model.py
git commit -m "Add new model architecture"

# Push to remote
git push origin feature/add-new-model

# Merge to main
git checkout main
git merge feature/add-new-model

# Tag releases
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Commit Message Convention
```
<type>(<scope>): <subject>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

Examples:
feat(model): add LSTM architecture
fix(preprocessing): handle missing values correctly
docs(readme): update installation instructions
```

---

## Python Environment Management

### venv
```bash
# Create environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Deactivate
deactivate

# Install requirements
pip install -r requirements.txt

# Freeze requirements
pip freeze > requirements.txt
```

### conda
```bash
# Create environment
conda create -n myenv python=3.9

# Activate
conda activate myenv

# Deactivate
conda deactivate

# Install packages
conda install numpy pandas scikit-learn

# From requirements.txt
pip install -r requirements.txt

# Export environment
conda env export > environment.yml

# Create from yml
conda env create -f environment.yml

# List environments
conda env list

# Remove environment
conda env remove -n myenv
```

### Requirements Files

**requirements.txt** (production):
```
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
mlflow==2.8.0
dvc==3.0.0
```

**requirements-dev.txt** (development):
```
-r requirements.txt
jupyter
pytest
black
flake8
mypy
```

### Install from requirements
```bash
# Basic
pip install -r requirements.txt

# With specific index
pip install -r requirements.txt -i https://pypi.org/simple

# Upgrade all
pip install -r requirements.txt --upgrade
```

---

## Quick Command Cheatsheet

### Start ML Project
```bash
mkdir my-ml-project && cd my-ml-project
git init
python -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn mlflow dvc
dvc init
git add .
git commit -m "Initial commit"
```

### Daily Workflow
```bash
# Pull latest code and data
git pull
dvc pull

# Work on code
# ...

# Track new data
dvc add data/new_dataset.csv
git add data/new_dataset.csv.dvc .gitignore
git commit -m "Add new dataset"
dvc push

# Run experiments
python src/train.py
mlflow ui  # View results

# Run full pipeline
dvc repro

# Commit changes
git add .
git commit -m "Update model"
git push
```

### Deploy
```bash
# Build Docker image
docker build -t my-ml-api:v1 .

# Run locally
docker run -p 8000:8000 my-ml-api:v1

# Or with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## Useful Links

- **DVC**: https://dvc.org/doc
- **MLflow**: https://mlflow.org/docs/
- **DagsHub**: https://dagshub.com/docs/
- **Hugging Face**: https://huggingface.co/docs
- **Ollama**: https://ollama.com/library
- **Docker**: https://docs.docker.com/
- **Transformers**: https://huggingface.co/docs/transformers/
- **PEFT**: https://huggingface.co/docs/peft/

---

This reference guide covers 90% of daily MLOps tasks. Bookmark and refer as needed!
