# ML-LLMOps: Complete Data Science & MLOps Project Repository

> A comprehensive repository covering end-to-end ML workflows, from data collection to deployment, with multiple approaches and tools.

## Overview

This repository serves as a complete reference for building production-ready machine learning systems. It covers:

- **Traditional ML** (Scikit-learn, XGBoost, LightGBM)
- **Deep Learning** (PyTorch, TensorFlow)
- **LLM Fine-tuning** (Hugging Face, LoRA, QLoRA)
- **MLOps Tools** (DVC, MLflow, Docker, DagsHub)
- **Deployment** (Flask, FastAPI, Streamlit, Ollama)

## Project Status

- ✅ DVC initialized for data versioning
- ✅ MLflow + DagsHub integration for experiment tracking
- ✅ Docker setup for containerization
- ✅ Basic ML pipeline example (ElasticNet Wine Quality)
- 📝 Comprehensive documentation for all workflows

## Documentation

### Core Guides

| Guide | Description | Status |
|-------|-------------|--------|
| [END_TO_END_ML_GUIDE.md](./END_TO_END_ML_GUIDE.md) | Complete ML lifecycle: data collection → deployment | ✅ Complete |
| [FINETUNING_GUIDE.md](./FINETUNING_GUIDE.md) | LLM fine-tuning with LoRA/QLoRA | ✅ Complete |
| [MLOPS_TOOLS_REFERENCE.md](./MLOPS_TOOLS_REFERENCE.md) | Quick reference for all MLOps tools | ✅ Complete |

### What Each Guide Covers

**END_TO_END_ML_GUIDE.md** - Your main reference for building ML projects:
- 9 phases of ML lifecycle with multiple approaches
- Data collection methods (APIs, web scraping, databases)
- EDA with pandas, visualization, automated tools
- Data preprocessing & feature engineering
- Model development (traditional ML, deep learning, LLMs)
- Experiment tracking with MLflow
- Deployment strategies (Flask, FastAPI, Docker, Cloud)
- Production monitoring
- Project templates and best practices

**FINETUNING_GUIDE.md** - Deep dive into LLM fine-tuning:
- What is fine-tuning and when to use it
- LoRA, QLoRA, Full Fine-tuning explained
- Mathematical concepts made simple
- Hyperparameter tuning guide
- Integration with DVC & MLflow
- Common pitfalls and solutions
- Practical decision trees

**MLOPS_TOOLS_REFERENCE.md** - Quick command reference:
- DVC commands and workflows
- MLflow Python API and CLI
- DagsHub integration
- Docker & docker-compose recipes
- Hugging Face transformers
- Ollama for local LLM inference
- Git best practices for ML

## Repository Structure

```
ML-LLMOps/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned data
├── src/
│   ├── mlflow/                 # MLflow examples
│   │   └── ElasticNetWineModel_Dagshub.py
│   └── MLmodel_In_Stages/      # Pipeline stages
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       ├── feature_engineering.py
│       ├── model_building.py
│       └── model_evaluation.py
├── models/                     # Saved models
├── notebooks/                  # Jupyter notebooks
├── docker-compose.yml          # Multi-container setup
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── dvc.yaml                    # DVC pipeline
├── params.yaml                 # Pipeline parameters
├── .dvc/                       # DVC configuration
└── .git/                       # Git version control
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/username/ML-LLMOps.git
cd ML-LLMOps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull data from DVC remote
dvc pull
```

### 2. Run Existing Example

```bash
# Run the wine quality model
python src/mlflow/ElasticNetWineModel_Dagshub.py

# View MLflow results
mlflow ui
# Open browser to http://localhost:5000
```

### 3. Explore with Docker

```bash
# Build and run services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Common Workflows

### Start a New ML Project

```bash
# Create project structure (see END_TO_END_ML_GUIDE.md for templates)
mkdir -p data/{raw,processed} src/{data,models,api} notebooks models reports/figures

# Initialize tools
git init
dvc init

# Track data
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add dataset"

# Setup remote storage (DagsHub example)
dvc remote add origin https://dagshub.com/USER/REPO.dvc
dvc push
```

### Run Experiments

```bash
# Method 1: Direct execution with MLflow
python src/models/train.py

# Method 2: DVC pipeline
dvc repro

# Method 3: With parameter changes
dvc exp run -S train.learning_rate=0.01 -S train.n_estimators=200

# View experiments
mlflow ui  # http://localhost:5000
```

### Fine-tune an LLM

See [FINETUNING_GUIDE.md](./FINETUNING_GUIDE.md) for detailed instructions.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Fine-tune...
```

### Deploy Model

```bash
# Option 1: Docker
docker build -t ml-api .
docker run -p 8000:8000 ml-api

# Option 2: Docker Compose
docker-compose up -d

# Option 3: Local (Flask/FastAPI)
python src/api/app.py
```

## Tools & Technologies

### Core Stack

- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, PyTorch, Transformers
- **MLOps**: DVC, MLflow, DagsHub
- **Deployment**: Docker, Flask, FastAPI, Ollama
- **Version Control**: Git

### Current Integrations

- **DagsHub**: Remote storage for data + MLflow tracking
- **Ollama**: Local LLM inference (configured in docker-compose.yml)
- **Docker**: Containerization for reproducible environments

## Examples in This Repo

### 1. Traditional ML: Wine Quality Prediction

**Location**: `src/mlflow/ElasticNetWineModel_Dagshub.py`

**What it does**:
- Predicts wine quality using ElasticNet regression
- Integrates with DagsHub for MLflow tracking
- Logs parameters, metrics, and models

**Run**:
```bash
python src/mlflow/ElasticNetWineModel_Dagshub.py
python src/mlflow/ElasticNetWineModel_Dagshub.py 0.5 0.6  # Custom alpha, l1_ratio
```

### 2. Pipeline Stages

**Location**: `src/MLmodel_In_Stages/`

Demonstrates breaking ML workflow into stages:
1. Data ingestion
2. Preprocessing
3. Feature engineering
4. Model building
5. Evaluation

Can be orchestrated with DVC pipeline (dvc.yaml).

## Learning Path

### Beginner
1. Read [END_TO_END_ML_GUIDE.md](./END_TO_END_ML_GUIDE.md) - Phase 1-3 (Data collection, EDA, Preprocessing)
2. Run the wine quality example
3. Practice with [MLOPS_TOOLS_REFERENCE.md](./MLOPS_TOOLS_REFERENCE.md) commands
4. Create your first DVC pipeline

### Intermediate
1. Complete all phases of END_TO_END_ML_GUIDE.md
2. Set up MLflow experiment tracking
3. Build and deploy a model with Docker
4. Implement monitoring and logging

### Advanced
1. Read [FINETUNING_GUIDE.md](./FINETUNING_GUIDE.md)
2. Fine-tune an LLM with LoRA
3. Deploy with Ollama
4. Set up complete CI/CD pipeline
5. Implement production monitoring

## Contributing

This repository serves as a learning resource and project template. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Resources

### Official Documentation
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Hugging Face Docs](https://huggingface.co/docs)
- [Ollama Models](https://ollama.com/library)

### Related Repositories
- [DagsHub ML Ops Guide](https://dagshub.com/docs/)
- [Hugging Face Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples)

## Current Setup

### Configured Tools

```yaml
# DVC
Remote: DagsHub
Status: Initialized ✅

# MLflow
Tracking URI: https://dagshub.com/nuthan.maddineni23/ML-LLMOps.mlflow
Status: Connected ✅

# Docker
Services: Ollama (LLM), custom ML API
Status: Configured ✅

# Git
Remote: GitHub/DagsHub
Status: Active ✅
```

### Dependencies (requirements.txt)

```
dagshub
mlflow<3
scikit-learn>=1.0
pandas
dvc
```

For LLM work, add:
```
transformers>=4.30.0
torch>=2.0.0
peft>=0.4.0
accelerate>=0.20.0
```

## Next Steps

Based on your goals, choose your path:

### Path 1: Traditional ML Focus
1. Explore more datasets
2. Build classification/regression models
3. Set up complete DVC pipeline
4. Deploy with Flask/FastAPI

### Path 2: Deep Learning Focus
1. Add PyTorch dependencies
2. Work with image/text datasets
3. Build neural networks
4. Use TensorBoard for monitoring

### Path 3: LLM Fine-tuning Focus
1. Add Hugging Face dependencies
2. Download base models
3. Prepare fine-tuning data
4. Fine-tune with LoRA
5. Deploy with Ollama

### Path 4: Full MLOps Stack
1. Implement all pipeline stages
2. Set up CI/CD
3. Add monitoring and alerts
4. Deploy to cloud (AWS/GCP/Azure)

## FAQ

**Q: Which guide should I read first?**
A: Start with END_TO_END_ML_GUIDE.md for overall understanding, then dive into specific guides based on your needs.

**Q: Do I need GPU for fine-tuning?**
A: For small models (<1B params) with LoRA, CPU works but is slow. For larger models, GPU is highly recommended. Use QLoRA for limited GPU memory.

**Q: How do I get base models?**
A: Use Hugging Face Hub (downloads automatically) or Ollama (for inference). See MLOPS_TOOLS_REFERENCE.md for details.

**Q: What's the difference between LoRA and RAG?**
A: LoRA modifies model weights (fine-tuning). RAG retrieves external information (no training). See FINETUNING_GUIDE.md intro.

**Q: Can I use this for production?**
A: Yes! The guides cover production deployment, monitoring, and best practices. Start simple and scale up.

## License

[Your License Here]

## Contact

[Your Contact Information]

---

**Happy ML Building! 🚀**

Start with any guide, explore the code, and build amazing ML systems!
