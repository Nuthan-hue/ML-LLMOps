# Model Finetuning: A Complete Learning Guide

## Table of Contents
1. [What is Finetuning?](#what-is-finetuning)
2. [Understanding Transfer Learning](#understanding-transfer-learning)
3. [Types of Finetuning Methods](#types-of-finetuning-methods)
4. [The Finetuning Workflow](#the-finetuning-workflow)
5. [Integration with DVC & MLflow](#integration-with-dvc--mlflow)
6. [Common Pitfalls](#common-pitfalls-to-avoid)
7. [Practical Decision Guide](#practical-decision-guide)

---

## What is Finetuning?

### Analogy
Think of finetuning like teaching a professional chef (pretrained model) to cook your grandmother's specific recipes. The chef already knows:
- Cooking techniques
- Ingredient combinations
- Kitchen skills
- Food chemistry

You're just teaching them your specific style and preferences.

### Key Concept
- **Pretrained Model**: Already learned general patterns from massive datasets
- **Finetuning**: Adapt it to your specific task/domain with smaller dataset

### Example Scenario
**Base Model**: GPT trained on internet text (knows general English)
**After Finetuning**: Understands medical terminology and can answer healthcare questions

### Why Finetuning Works
```
General Knowledge (pretrained)
    ↓
  [Your specific data]
    ↓
Specialized Knowledge (finetuned)
```

The model doesn't start from scratch - it already has foundational knowledge and you're adding specialization.

---

## Understanding Transfer Learning

### The Foundation of Finetuning

**Transfer Learning** = Using knowledge from one task to help with another

### How Neural Networks Learn in Layers

```
Input → Layer 1 → Layer 2 → Layer 3 → ... → Output

Layer 1: Basic patterns (edges, simple shapes, common words)
Layer 2: Medium patterns (textures, phrases, syntax)
Layer 3: Complex patterns (objects, semantics, context)
...
Final Layers: Task-specific features
```

### Key Insight
- **Lower layers**: Learn general features (useful for ANY task)
- **Higher layers**: Learn task-specific features (specific to training data)

### Finetuning Strategy
You have two main choices:
1. **Freeze lower layers** → Only train higher layers
2. **Train all layers** → But with lower learning rates for lower layers

---

## Types of Finetuning Methods

### Method Comparison Table

| Method | Parameters Trained | Memory Required | Performance | Use Case |
|--------|-------------------|-----------------|-------------|----------|
| Full Finetuning | 100% | Very High | Best | Large dataset, lots of compute |
| LoRA | ~0.1-1% | Low | 90% of full | Most common, great balance |
| QLoRA | ~0.1-1% | Very Low | 85% of full | Limited GPU memory |
| Adapter Tuning | ~2-5% | Medium | Good | Modular task switching |

---

### 1. Full Finetuning

#### What Happens
Update **ALL** model parameters during training.

#### Analogy
Retraining the entire chef from scratch on your recipes.

#### Technical Details
- Every weight in every layer gets updated
- Requires storing gradients for all parameters
- Memory requirement = Model size × 4
  - Forward pass activations
  - Backward pass gradients
  - Optimizer states (momentum, variance)
  - Parameter updates

#### Example
```
Model: 7B parameters
Memory needed: 7B × 4 bytes × 4 = 112 GB

Why ×4?
- Model weights: 7B × 4 bytes = 28GB (float32)
- Gradients: 28GB
- Optimizer state (Adam): 56GB (2× for momentum + variance)
Total: 112GB
```

#### Pros
- Best possible performance
- Full control over all parameters
- Can completely change model behavior

#### Cons
- Needs lots of memory
- Expensive (time and compute)
- Risk of catastrophic forgetting
- Risk of overfitting on small datasets

#### When to Use
- Large dataset (10K+ examples)
- Lots of compute resources
- Task very different from pretraining
- Need maximum performance

---

### 2. LoRA (Low-Rank Adaptation) ⭐ Most Popular

#### Core Concept
Instead of changing the huge weight matrices, add small "adapter" matrices.

#### Analogy
Instead of retraining the chef, give them a small recipe card to reference while cooking.

#### Mathematical Explanation

**Original forward pass:**
```
Y = W × X

Where:
- W: Weight matrix (e.g., 4096 × 4096)
- X: Input vector
- Y: Output vector
```

**With LoRA:**
```
Y = W × X + (B × A) × X

Where:
- W: Frozen pretrained weights (NOT trained)
- B: Small matrix (4096 × r)
- A: Small matrix (r × 4096)
- r: Rank (typically 4-64)
- B × A: Creates a low-rank update to W
```

#### Why This is Genius

**Parameter calculation example:**
```
Original: W has 4096 × 4096 = 16,777,216 parameters

With LoRA (r=8):
- B: 4096 × 8 = 32,768 parameters
- A: 8 × 4096 = 32,768 parameters
- Total: 65,536 parameters (0.39% of original!)
```

**You train only 0.4% of parameters but get ~90% of full finetuning performance!**

#### Key Parameters

**1. Rank (r)**
- Controls the capacity of the adapter
- Higher rank = more expressive, more memory
- Lower rank = faster, less memory, might underfit

```
Typical values:
- r = 4: Very light, fast
- r = 8: Most common starting point
- r = 16: Higher capacity
- r = 32-64: Complex tasks
```

**2. Alpha (α)**
- Scaling factor for LoRA updates
- Usually set to 2× the rank
- Higher alpha = stronger LoRA effect

```
Typical: alpha = 2 × r
If r=8, then alpha=16
```

**3. Target Modules**
- Which layers to apply LoRA to
- Common: Attention layers (query, key, value projections)
- Can also target feedforward layers

```
For transformers:
- q_proj: Query projection
- v_proj: Value projection
- k_proj: Key projection (sometimes)
- o_proj: Output projection (sometimes)
```

**4. Dropout**
- Regularization for LoRA layers
- Prevents overfitting
- Typical: 0.05 - 0.1

#### Pros
- Memory efficient (10-100× less than full finetuning)
- Fast training
- Multiple LoRA adapters can be trained for different tasks
- Easy to switch between tasks (swap adapters)
- Reduced overfitting risk

#### Cons
- Slightly lower performance than full finetuning
- Limited by rank bottleneck
- Not suitable if task is very different from pretraining

#### When to Use
- **Most common method** - start here!
- Limited GPU memory
- Multiple tasks to finetune for
- Dataset: 100-10K examples
- Task somewhat related to pretraining

---

### 3. QLoRA (Quantized LoRA)

#### Concept
LoRA + Quantization of the base model

#### How It Works
```
1. Load pretrained model in 4-bit precision (instead of 16-bit or 32-bit)
2. Keep LoRA adapters in full precision (16-bit)
3. Train only the adapters (like regular LoRA)
4. During forward pass:
   - Dequantize weights to perform computation
   - Apply LoRA updates
   - Compute gradients only for LoRA
```

#### Quantization Explained

**Normal model storage (FP16):**
```
Each parameter = 16 bits = 2 bytes
7B model = 7B × 2 bytes = 14 GB
```

**Quantized model (4-bit):**
```
Each parameter = 4 bits = 0.5 bytes
7B model = 7B × 0.5 bytes = 3.5 GB
```

**Memory savings: 4× reduction!**

#### Why It's Revolutionary
Can finetune massive models on consumer hardware:
- 65B parameter model on single 48GB GPU
- 13B parameter model on 16GB GPU
- 7B parameter model on 8GB GPU

#### Trade-offs
- Base model slightly less accurate (quantization loss)
- Slower than regular LoRA (dequantization overhead)
- Final performance ~85-95% of full-precision LoRA
- But enables finetuning otherwise impossible models!

#### When to Use
- Very limited GPU memory
- Want to finetune large models (30B+)
- Can accept slight quality trade-off
- Can't afford multiple GPUs

---

### 4. Adapter Tuning

#### Concept
Insert small neural network modules (adapters) between existing layers.

#### Architecture
```
Layer 1 (frozen)
    ↓
[Adapter Module] ← Trainable
    ↓
Layer 2 (frozen)
    ↓
[Adapter Module] ← Trainable
    ↓
Layer 3 (frozen)
```

#### Adapter Module Structure
```
Input
  ↓
Down-projection (reduce dimensions)
  ↓
Non-linearity (ReLU/GELU)
  ↓
Up-projection (restore dimensions)
  ↓
Residual connection (add to input)
  ↓
Output
```

#### Difference from LoRA

**LoRA:**
- Modifies layer computation (parallel path)
- W×X + LoRA×X
- Weights can be merged into base model

**Adapters:**
- Adds new layers in sequence
- Separate modules between layers
- Cannot be merged, always separate

#### Pros
- Modular (easy to add/remove)
- Can have different adapters for different tasks
- More structured than LoRA

#### Cons
- Adds computational overhead (more layers)
- Slightly more parameters than LoRA
- Slower inference than merged LoRA

#### When to Use
- Need to switch between many tasks
- Want strict modularity
- Can afford slight inference slowdown

---

## The Finetuning Workflow

### Step 1: Data Preparation

#### What You Need
Input-output pairs for your specific task.

#### Format Examples

**Text Generation / Instruction Following:**
```json
{
  "instruction": "Translate the following to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**Text Classification:**
```json
{
  "text": "This movie was absolutely fantastic!",
  "label": "positive"
}
```

**Question Answering:**
```json
{
  "question": "What is the capital of France?",
  "context": "France is a country in Europe. Its capital city is Paris.",
  "answer": "Paris"
}
```

**Chat/Dialogue:**
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have access to real-time weather data."}
  ]
}
```

#### Data Quality > Quantity

**Better to have:**
- 500 high-quality, accurate examples
- Clean, consistent format
- Representative of real use cases

**Than:**
- 10,000 noisy examples
- Inconsistent formatting
- Not representative

#### Dataset Size Guidelines

```
< 100 examples: Too small, try few-shot prompting instead
100-500: Minimum for LoRA, expect modest improvements
500-2,000: Good for LoRA, solid results
2,000-10,000: Great for LoRA, excellent for light full finetuning
10,000+: Can do full finetuning
```

#### Train/Val/Test Splits

**Standard split:**
- Train: 80% (model learns from this)
- Validation: 10% (monitor during training, tune hyperparameters)
- Test: 10% (final evaluation, NEVER seen during training)

**Why validation set matters:**
```
Epoch 1: Train loss = 0.8, Val loss = 0.85
Epoch 2: Train loss = 0.6, Val loss = 0.7
Epoch 3: Train loss = 0.4, Val loss = 0.65
Epoch 4: Train loss = 0.2, Val loss = 0.75 ← STOP! Overfitting!
```

**Key principle:** Test set is your final exam - don't look at it until the very end!

---

### Step 2: Choose Base Model

#### Considerations

**1. Task Alignment**

| Your Task | Recommended Model Type |
|-----------|------------------------|
| Text generation | GPT-style (GPT-2, GPT-J, LLaMA) |
| Text understanding | BERT-style (BERT, RoBERTa, DeBERTa) |
| Both | T5, BART |
| Code | CodeGen, StarCoder, CodeLLaMA |
| Vision | ResNet, ViT, CLIP |
| Multimodal | CLIP, Flamingo, LLaVA |

**2. Model Size vs Resources**

```
Small (< 1B parameters):
- Examples: BERT-base (110M), DistilGPT-2 (82M)
- Memory: Can run on 8GB GPU
- Training time: Hours
- Use when: Quick experiments, limited resources

Medium (1-13B parameters):
- Examples: GPT-2 (1.5B), LLaMA-7B (7B), Flan-T5-XL (3B)
- Memory: 16-24GB GPU
- Training time: Days
- Use when: Production, good balance

Large (13-70B parameters):
- Examples: LLaMA-65B, GPT-3
- Memory: Multiple GPUs or QLoRA
- Training time: Weeks
- Use when: Need best performance, have resources
```

**3. Licensing**

Check if you can use it for your purpose:
- **Commercial-friendly:** LLaMA 2, Falcon, MPT
- **Research only:** LLaMA 1 (was research-only)
- **Open source:** GPT-2, BERT, T5

**4. Domain Specialization**

Start with domain-specific base if available:
- Medical: BioBERT, BioGPT, Med-PaLM
- Code: CodeBERT, CodeGen, StarCoder
- Science: SciBERT, Galactica
- Finance: FinBERT

---

### Step 3: Configuration & Hyperparameters

#### Critical Hyperparameters Explained

**1. Learning Rate (Most Important!)**

**What it is:** Size of the steps during training

**Analogy:** Walking down a hill to find the lowest point
- Too large: Take huge steps, jump over the minimum, unstable
- Too small: Tiny steps, takes forever, might get stuck
- Just right: Efficient progress to minimum

**Typical values:**
```
Full finetuning: 1e-5 to 5e-5 (0.00001 to 0.00005)
LoRA: 1e-4 to 3e-4 (0.0001 to 0.0003)
QLoRA: 2e-4 to 5e-4

Why higher for LoRA?
- Only training small adapters
- Can be more aggressive
- Less risk of catastrophic forgetting
```

**How to find the right one:**
1. Start with recommended value
2. If loss doesn't decrease: Try higher learning rate
3. If loss explodes/NaN: Lower learning rate
4. Use learning rate finder or schedulers

---

**2. Batch Size**

**What it is:** How many examples to process before updating weights

**Analogy:**
- Batch size = 1: Update after each student's test (noisy)
- Batch size = 32: Update after 32 students (smoother average)

**Trade-offs:**
```
Small batch (1-8):
+ Less memory
+ More updates per epoch (can be beneficial)
- Noisy gradients
- Unstable training

Large batch (32-128):
+ Stable gradients
+ Better GPU utilization
- More memory
- Fewer updates per epoch
```

**Memory constraint workaround: Gradient Accumulation**
```
Actual batch size = 4 (fits in memory)
Gradient accumulation steps = 8
Effective batch size = 4 × 8 = 32

How it works:
1. Process 4 examples, accumulate gradients (don't update)
2. Process 4 more, accumulate gradients
3. ... repeat 8 times
4. Update weights with accumulated gradients
```

---

**3. Number of Epochs**

**What it is:** How many times to go through the entire dataset

**Analogy:** Reading a textbook
- 1 pass: Might miss things
- 3-5 passes: Good understanding
- 20 passes: Memorized but might not truly understand (overfitting)

**Guidelines:**
```
Small dataset (< 1000 examples): 5-10 epochs
Medium dataset (1000-10000): 3-5 epochs
Large dataset (> 10000): 1-3 epochs

Watch validation loss:
- Decreasing: Keep training
- Plateaued: Almost done
- Increasing: STOP, overfitting!
```

---

**4. Warmup Steps**

**What it is:** Gradually increase learning rate at the start

**Why it helps:**
- Cold start: Model weights are pretrained but new task
- Sudden large updates can destabilize training
- Warmup eases the transition

**Visual:**
```
Learning Rate over time:

      |           _______________  ← Full LR
      |          /
      |         /
      |        /
      |_______/
      |
      0    warmup    training
```

**Typical:** 5-10% of total training steps

---

**5. Weight Decay**

**What it is:** Regularization that penalizes large weights

**Purpose:** Prevent overfitting

**Typical value:** 0.01

---

**6. Max Sequence Length**

**What it is:** Maximum number of tokens to process

**Trade-offs:**
```
Shorter (256-512):
+ Faster training
+ Less memory
- Might truncate important context

Longer (1024-2048):
+ Full context
- Slower, more memory
```

**Guideline:** Analyze your data, use 95th percentile length

---

### Step 4: Training Loop (What Actually Happens)

#### The Training Process Explained

```
For each epoch (1 to num_epochs):

    For each batch in training_data:

        1. FORWARD PASS
           - Feed batch through model
           - Get predictions

        2. CALCULATE LOSS
           - Compare predictions to true labels
           - Compute error (loss)

        3. BACKWARD PASS
           - Calculate gradients (how to adjust weights)
           - Backpropagation through the network

        4. UPDATE WEIGHTS
           - Adjust trainable parameters
           - Use optimizer (Adam, AdamW, etc.)

        5. LOG METRICS
           - Record loss, learning rate
           - Send to MLflow

    After each epoch:

        6. VALIDATION
           - Evaluate on validation set
           - Calculate validation loss & metrics

        7. CHECKPOINT
           - If best validation loss:
               - Save model checkpoint
               - Mark as best model

        8. EARLY STOPPING CHECK
           - If validation loss hasn't improved for N epochs:
               - Stop training
               - Prevent overfitting
```

#### Key Metrics to Monitor

**During Training:**

1. **Training Loss**
   - Should steadily decrease
   - If stuck: Increase learning rate
   - If exploding: Decrease learning rate

2. **Validation Loss**
   - Should decrease with training loss
   - If increasing while training decreases: OVERFITTING!

3. **Learning Rate**
   - Track if using schedulers
   - Helps debug training issues

4. **Gradient Norm**
   - Measure of gradient magnitude
   - Very high: Might need gradient clipping
   - Very low: Dead neurons or vanishing gradients

**Healthy Training Pattern:**
```
Epoch | Train Loss | Val Loss | Status
------|------------|----------|--------
  1   |   2.45     |  2.50    | ✓ Both decreasing
  2   |   1.82     |  1.89    | ✓ Good progress
  3   |   1.23     |  1.35    | ✓ Still improving
  4   |   0.95     |  1.32    | ⚠ Val plateauing
  5   |   0.71     |  1.38    | ❌ OVERFITTING! Stop here
```

**Unhealthy Patterns:**
```
Pattern 1: Underfitting
Epoch | Train Loss | Val Loss
  1   |   2.45     |  2.48
  2   |   2.40     |  2.44
  3   |   2.38     |  2.42
→ Not learning enough, increase capacity or learning rate

Pattern 2: Overfitting from start
Epoch | Train Loss | Val Loss
  1   |   1.20     |  2.80
  2   |   0.50     |  3.20
→ Memorizing training set, need regularization

Pattern 3: Unstable
Epoch | Train Loss | Val Loss
  1   |   2.40     |  2.45
  2   |   3.80     |  3.90
  3   |   1.20     |  1.80
→ Learning rate too high or bad data
```

---

### Step 5: Evaluation

#### Comparing Models

**You must compare:**
1. **Baseline:** Pretrained model (no finetuning)
2. **Finetuned:** Your finetuned model

**This tells you if finetuning actually helped!**

#### Metrics by Task Type

**Classification Tasks:**
```
Accuracy: What % did we get right?
  - Simple, intuitive
  - Can be misleading with imbalanced data

Precision: Of predicted positives, how many were correct?
  - High precision = Few false positives

Recall: Of actual positives, how many did we find?
  - High recall = Few false negatives

F1 Score: Harmonic mean of precision and recall
  - Balances both metrics
  - Good for imbalanced datasets
```

**Generation Tasks:**
```
Perplexity: How "surprised" is the model?
  - Lower = Better
  - Measures how well model predicts next token

BLEU: Overlap with reference (translation)
  - 0-100 scale
  - Higher = Better

ROUGE: Recall-oriented (summarization)
  - Measures overlap with reference summary

Human Evaluation: Gold standard
  - Ask humans to rate outputs
  - Expensive but most reliable
```

**Question Answering:**
```
Exact Match: % of questions with perfect answer

F1 Score: Token overlap between prediction and answer

METEOR: Considers synonyms and stems
```

#### Evaluation Best Practices

**1. Multiple metrics:** Don't rely on just one

**2. Qualitative analysis:** Look at actual predictions
```
Good metric but bad outputs?
- Model gaming the metric
- Metric doesn't capture what matters

Bad metric but good outputs?
- Wrong metric for your use case
- Need human evaluation
```

**3. Error analysis:**
```
Categories of errors:
- What types of examples fail?
- Are there patterns?
- Can you collect more data for weak areas?
```

**4. A/B testing:**
```
If deploying:
- Show base model to 50% of users
- Show finetuned to 50%
- Measure real-world metrics (clicks, satisfaction, etc.)
```

---

## Integration with DVC & MLflow

### How DVC Fits Into Finetuning

#### What is DVC?
Data Version Control - Git for data and ML pipelines

#### Key Concepts

**1. Pipeline as a DAG (Directed Acyclic Graph)**
```
Your workflow as stages:

data_ingestion → data_preprocessing → prepare_finetuning
                                             ↓
                                      finetune_model
                                             ↓
                                      evaluate_model
```

**2. Dependency Tracking**

Each stage in `dvc.yaml`:
```yaml
stage_name:
  cmd: python script.py           # What to run
  deps:                            # Inputs
    - input_file.csv
    - script.py
  outs:                            # Outputs
    - output_file.csv
  params:                          # Parameters from params.yaml
    - learning_rate
```

**How it works:**
- DVC tracks checksums of dependencies
- If dependency changes → stage is "dirty" → needs rerun
- If nothing changed → use cached output → skip stage

**3. Benefits for Finetuning**

**Reproducibility:**
```
Someone else can run:
$ dvc repro

And get exact same results:
- Same data
- Same code
- Same parameters
- Same model
```

**Versioning:**
```
$ git checkout experiment-lora-r8
$ dvc checkout

Now you have:
- Code from that experiment
- Data from that experiment
- Model from that experiment
```

**Sharing:**
```
$ dvc push   # Upload data/models to remote storage
$ dvc pull   # Download data/models

Git: Code and configs
DVC: Data and models
```

#### Example dvc.yaml for Finetuning

```yaml
stages:

  # Stage 1: Get raw data
  data_ingestion:
    cmd: python src/data/data_ingestion.py
    deps:
      - src/data/data_ingestion.py
    outs:
      - data/raw/train.csv
      - data/raw/test.csv

  # Stage 2: Clean and preprocess
  data_preprocessing:
    cmd: python src/data/data_preprocessing.py
    deps:
      - data/raw/train.csv
      - data/raw/test.csv
      - src/data/data_preprocessing.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  # Stage 3: Format for finetuning
  prepare_finetuning:
    cmd: python src/data/prepare_finetuning_data.py
    deps:
      - data/processed/train.csv
      - data/processed/test.csv
      - src/data/prepare_finetuning_data.py
    outs:
      - data/finetuning/train.jsonl
      - data/finetuning/val.jsonl
      - data/finetuning/test.jsonl
    params:
      - prepare.train_ratio
      - prepare.val_ratio

  # Stage 4: Finetune model
  finetune_model:
    cmd: python src/models/finetune.py
    deps:
      - data/finetuning/train.jsonl
      - data/finetuning/val.jsonl
      - src/models/finetune.py
      - configs/finetuning/lora_config.yaml
    outs:
      - models/checkpoints:
          persist: true  # Keep this even after stage reruns
    params:
      - finetune.learning_rate
      - finetune.batch_size
      - finetune.num_epochs
      - finetune.lora_r
      - finetune.lora_alpha

  # Stage 5: Evaluate
  evaluate_model:
    cmd: python src/evaluation/evaluate.py
    deps:
      - models/checkpoints
      - data/finetuning/test.jsonl
      - src/evaluation/evaluate.py
    metrics:
      - metrics.json:
          cache: false  # Always regenerate, don't cache
    plots:
      - plots/confusion_matrix.png
```

#### params.yaml

```yaml
# Data preparation
prepare:
  train_ratio: 0.8
  val_ratio: 0.1
  max_length: 512

# Finetuning
finetune:
  model_name: "distilbert-base-uncased"
  method: "lora"

  # Training hyperparameters
  learning_rate: 2e-4
  batch_size: 8
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01

  # LoRA config
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "v_proj"]

# Evaluation
evaluation:
  metrics: ["accuracy", "f1", "precision", "recall"]
```

#### DVC Commands for Finetuning

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro finetune_model

# Show pipeline visualization
dvc dag

# See what changed
dvc status

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff experiment1 experiment2

# Push data/models to remote
dvc push

# Pull data/models from remote
dvc pull
```

---

### How MLflow Fits Into Finetuning

#### What is MLflow?
Experiment tracking and model registry platform

#### Key Concepts

**1. Experiments**
Collection of related runs (e.g., "bert-finetuning-sentiment")

**2. Runs**
Single execution with:
- Parameters (hyperparameters)
- Metrics (loss, accuracy over time)
- Artifacts (models, plots, configs)
- Code version (git commit)

**3. Model Registry**
Versioned model storage with stages (Staging, Production)

#### What to Log in MLflow

**Parameters (hyperparameters):**
```python
mlflow.log_param("learning_rate", 2e-4)
mlflow.log_param("batch_size", 8)
mlflow.log_param("lora_r", 8)
mlflow.log_param("lora_alpha", 16)
mlflow.log_param("num_epochs", 3)
mlflow.log_param("model_name", "distilbert-base-uncased")
```

**Metrics (during training):**
```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss, val_acc = validate()

    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
```

**Artifacts (files):**
```python
# Save model
mlflow.pytorch.log_model(model, "finetuned_model")

# Save configs
mlflow.log_artifact("configs/finetuning/lora_config.yaml")

# Save plots
mlflow.log_artifact("plots/training_curve.png")

# Save tokenizer
mlflow.log_artifact("tokenizer/")
```

**System metrics:**
```python
mlflow.log_param("gpu_type", "NVIDIA A100")
mlflow.log_param("gpu_memory", "40GB")
mlflow.set_tag("git_commit", git_commit_hash)
```

#### Example MLflow Integration

```python
import mlflow
import mlflow.pytorch
from transformers import AutoModelForSequenceClassification, TrainingArguments

# Set experiment
mlflow.set_experiment("sentiment-classification-finetuning")

# Start run
with mlflow.start_run(run_name="lora-r8-lr2e4"):

    # Log all parameters
    mlflow.log_params({
        "model_name": "distilbert-base-uncased",
        "method": "lora",
        "learning_rate": 2e-4,
        "batch_size": 8,
        "num_epochs": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "dataset_size": len(train_dataset)
    })

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch()
        val_metrics = validate()

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"]
        }, step=epoch)

        # Save checkpoint
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(f"checkpoints/epoch_{epoch}")

    # Final evaluation
    test_metrics = evaluate_on_test()
    mlflow.log_metrics({
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"]
    })

    # Log model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="sentiment-classifier"
    )

    # Log artifacts
    mlflow.log_artifact("configs/lora_config.yaml")
    mlflow.log_artifact("plots/confusion_matrix.png")
```

#### MLflow UI Benefits

**Compare runs side-by-side:**
```
Run 1: lora_r=4,  lr=1e-4  → F1: 0.85
Run 2: lora_r=8,  lr=2e-4  → F1: 0.89  ← Best!
Run 3: lora_r=16, lr=2e-4  → F1: 0.87
Run 4: lora_r=8,  lr=5e-4  → F1: 0.82
```

**Visualize metrics:**
- Training/validation curves
- Compare across runs
- Identify overfitting

**Model versioning:**
```
sentiment-classifier:
  Version 1: Initial baseline
  Version 2: With LoRA r=8  ← Staging
  Version 3: With LoRA r=16 ← Production
```

---

### DVC + MLflow: Better Together

**Separation of concerns:**

| Aspect | Tool | Purpose |
|--------|------|---------|
| Code version | Git | Track code changes |
| Data version | DVC | Track datasets, reproducibility |
| Experiment tracking | MLflow | Compare hyperparameters, metrics |
| Model registry | MLflow | Version models, deploy to production |
| Pipeline | DVC | Reproducible execution |

**Typical workflow:**
```
1. Write code → Git commit
2. Define pipeline → dvc.yaml
3. Run experiment → dvc repro
   └─ Inside stages → MLflow logging
4. Compare experiments → MLflow UI
5. Choose best model → MLflow model registry
6. Version everything → Git + DVC tags
7. Share with team → git push + dvc push
```

---

## Common Pitfalls to Avoid

### 1. Overfitting

#### What It Is
Model memorizes training data instead of learning generalizable patterns.

#### Signs
```
Training loss: 0.05 (very low)
Validation loss: 2.30 (high)

Model performs great on training data, terrible on new data.
```

#### Causes
- Too many epochs
- Model too large for dataset size
- Learning rate too high
- No regularization

#### Solutions
```
✓ Early stopping: Stop when validation loss increases
✓ Regularization: Weight decay, dropout
✓ More data: Collect more training examples
✓ Data augmentation: Create variations of existing data
✓ Reduce model capacity: Smaller LoRA rank, freeze more layers
✓ Lower learning rate: Take smaller steps
```

---

### 2. Catastrophic Forgetting

#### What It Is
Model forgets pretrained knowledge while learning new task.

#### Signs
```
Before finetuning:
  General question: "What is 2+2?" → "4" ✓

After finetuning on medical QA:
  Medical question: "What is diabetes?" → Good answer ✓
  General question: "What is 2+2?" → Nonsense ✗
```

#### Causes
- Learning rate too high
- Too many epochs
- Task very different from pretraining

#### Solutions
```
✓ Use LoRA instead of full finetuning
✓ Lower learning rate (1e-5 vs 1e-4)
✓ Fewer epochs
✓ Mix general data with task-specific data
✓ Freeze more layers (only train top layers)
```

---

### 3. Data Leakage

#### What It Is
Test data "leaks" into training, inflating performance metrics.

#### Common Scenarios

**Scenario 1: Duplicate data**
```
Training set: ["This movie is great!", ...]
Test set: ["This movie is great!", ...]  ← Same example!
```

**Scenario 2: Temporal leakage**
```
Task: Predict stock prices
Training: 2020-2022 data
Test: 2021 data  ← Overlaps with training!

Should be:
Training: 2020-2021 data
Test: 2022 data (strictly future)
```

**Scenario 3: Feature leakage**
```
Predicting customer churn
Feature: "days_since_last_login"

But this is calculated AFTER churn happened!
```

#### Solutions
```
✓ Strict temporal split for time-series
✓ Check for duplicates (hash each example)
✓ Holdout test set before any exploration
✓ Be careful with data augmentation (don't augment test set)
✓ Never look at test set until final evaluation
```

---

### 4. Hardware Limitations / OOM (Out of Memory)

#### Signs
```
RuntimeError: CUDA out of memory
OR
Killed (process runs out of RAM)
```

#### Solutions

**Immediate fixes:**
```
1. Reduce batch size:
   batch_size = 32 → 16 → 8 → 4

2. Use gradient accumulation:
   batch_size = 4
   accumulation_steps = 8
   effective_batch = 32

3. Reduce sequence length:
   max_length = 1024 → 512 → 256

4. Use gradient checkpointing:
   Trade compute for memory
   Slower but uses less memory
```

**Bigger changes:**
```
5. Use LoRA instead of full finetuning
   100× less memory

6. Use QLoRA instead of LoRA
   4× less memory than LoRA

7. Use mixed precision (FP16/BF16)
   2× less memory

8. Use smaller model
   GPT-3 → GPT-2
   BERT-large → BERT-base → DistilBERT

9. Use model parallelism
   Split model across multiple GPUs

10. Use DeepSpeed/FSDP
    Advanced optimization frameworks
```

---

### 5. Poor Metric Selection

#### Problem
Optimizing for wrong metric leads to "good numbers, bad model"

#### Examples

**Example 1: Imbalanced classification**
```
Dataset: 95% negative, 5% positive

Model that always predicts "negative":
  Accuracy: 95%  ← Looks great!
  But useless for finding positives

Better metric: F1 score, Precision-Recall AUC
```

**Example 2: Generation quality**
```
Translation task

Model learns to copy input:
  BLEU score: High (overlaps with reference)
  But translation is wrong!

Better: Add human evaluation
```

**Example 3: Real-world impact**
```
Click prediction model

Model A: 85% accuracy, balanced errors
Model B: 88% accuracy, but misses important edge cases

Model A might be better for production!
```

#### Solutions
```
✓ Use multiple metrics
✓ Include task-specific metrics
✓ Do qualitative analysis (look at predictions)
✓ Consider business metrics, not just ML metrics
✓ Get human evaluation for generation tasks
```

---

### 6. Insufficient Training

#### Signs
```
Both training and validation loss are still decreasing
Model hasn't learned the task well
Performance far below baseline
```

#### Causes
- Too few epochs
- Learning rate too low
- Model too small
- Insufficient data

#### Solutions
```
✓ Train for more epochs (watch validation loss)
✓ Increase learning rate
✓ Use larger LoRA rank or full finetuning
✓ Collect more/better data
✓ Check data quality
```

---

### 7. Forgetting to Version/Track Experiments

#### Problem
```
"I had a model that worked great last week!"
"What hyperparameters did I use?"
"Where is that checkpoint?"
```

#### Solutions
```
✓ Use MLflow for every experiment
✓ Use DVC to version data and models
✓ Git commit before each experiment
✓ Name experiments descriptively
✓ Document what you try
```

---

## Practical Decision Guide

### Should You Finetune?

#### Decision Tree

```
Start: I have a task for an LLM

Q1: Can you solve it with prompting?
    ├─ Yes → Use prompting (zero-shot/few-shot)
    │         ✓ Faster, cheaper, no training needed
    │
    └─ No → Q2

Q2: Do you have < 100 quality examples?
    ├─ Yes → Try few-shot prompting or collect more data
    │         Finetuning unlikely to work well
    │
    └─ No → Q3

Q3: Do you need faster inference or lower costs?
    ├─ Yes → Consider finetuning smaller model
    │         Small finetuned model > Large prompted model
    │
    └─ No → Q4

Q4: Is your domain very specialized?
    ├─ Yes → Finetune! (Medical, legal, etc.)
    │
    └─ No → Q5

Q5: Do you need consistent output format?
    ├─ Yes → Finetune! (More reliable than prompting)
    │
    └─ Maybe try prompting first

```

---

### Choosing Finetuning Method

```
Start: I'm going to finetune

Q1: Do you have < 2000 examples?
    ├─ Yes → LoRA
    └─ No → Q2

Q2: GPU memory > 40GB?
    ├─ Yes → Can try full finetuning
    └─ No → Q3

Q3: GPU memory > 16GB?
    ├─ Yes → LoRA
    └─ No → QLoRA

Q4: Need to finetune for multiple tasks?
    ├─ Yes → LoRA (train separate adapters)
    └─ No → Continue

Q5: Is task VERY different from pretraining?
    ├─ Yes → Full finetuning (if possible)
    └─ No → LoRA is fine

RECOMMENDATION: Start with LoRA, it works 90% of the time!
```

---

### Hyperparameter Selection Guide

#### First-time finetuning? Start here:

```yaml
# LoRA configuration
lora_r: 8                    # Good starting point
lora_alpha: 16               # 2× rank
lora_dropout: 0.05           # Light regularization
target_modules: ["q_proj", "v_proj"]  # Standard for transformers

# Training
learning_rate: 2e-4          # LoRA default
batch_size: 8                # Adjust based on memory
num_epochs: 3                # Usually sufficient
warmup_steps: 100            # ~10% of total steps
weight_decay: 0.01           # Regularization

# Data
max_length: 512              # Balance of context and speed
train_val_split: 0.9         # 90% train, 10% val
```

#### If results are bad:

**Underfitting (not learning enough):**
```
→ Increase learning_rate: 2e-4 → 3e-4
→ Increase lora_r: 8 → 16
→ Increase num_epochs: 3 → 5
→ Check data quality
```

**Overfitting (memorizing training data):**
```
→ Decrease learning_rate: 2e-4 → 1e-4
→ Decrease lora_r: 8 → 4
→ Decrease num_epochs: 3 → 2
→ Increase dropout: 0.05 → 0.1
→ Collect more data
```

**Unstable training (loss spiking):**
```
→ Decrease learning_rate: 2e-4 → 1e-4
→ Increase warmup_steps: 100 → 500
→ Check for bad data (corrupted examples)
→ Use gradient clipping
```

---

### Dataset Size Guidelines

```
| Examples | Finetuning Method | Expected Results |
|----------|-------------------|------------------|
| < 100    | Don't finetune    | Try few-shot prompting |
| 100-500  | LoRA (r=4-8)      | Modest improvement |
| 500-2K   | LoRA (r=8-16)     | Good improvement |
| 2K-10K   | LoRA (r=16-32)    | Great results |
| 10K-100K | Full or LoRA      | Excellent results |
| > 100K   | Full finetuning   | Best possible |
```

---

### Computational Resources Guide

```
| GPU Memory | Model Size | Method | Batch Size |
|------------|------------|--------|------------|
| 8 GB       | < 1B       | LoRA   | 2-4        |
| 16 GB      | < 3B       | LoRA   | 4-8        |
| 24 GB      | < 7B       | LoRA   | 8-16       |
| 40 GB      | < 13B      | LoRA   | 16-32      |
| 40 GB      | 30B-70B    | QLoRA  | 4-8        |
| 80 GB      | < 70B      | LoRA   | 8-16       |
```

**No GPU?**
- Use Google Colab (free T4 GPU)
- Use Kaggle Notebooks (free GPU)
- Use cloud (AWS, GCP, Azure)
- Use smaller models (CPU fine for <500M params)

---

## Next Steps: Your Learning Path

### Phase 1: Understand (Current)
- ✓ Read this guide
- Learn the concepts
- Understand trade-offs

### Phase 2: Simple Example
```
Task: Sentiment classification (positive/negative reviews)
Dataset: 1000 examples
Model: DistilBERT (small, fast)
Method: LoRA
Goal: Get hands-on experience
```

**What you'll learn:**
- Data preparation
- Basic training loop
- Evaluation
- Using Hugging Face libraries

### Phase 3: Your Use Case
```
Task: Your specific problem
Dataset: Your data
Model: Appropriate for task
Method: Based on resources
```

**What you'll learn:**
- Data quality matters
- Hyperparameter tuning
- Debugging training issues
- Real-world challenges

### Phase 4: Advanced Topics
```
- Advanced techniques (LoRA+, DoRA)
- Multi-task learning
- Continual learning
- Deployment and serving
- Monitoring in production
```

---

## Questions to Guide Your Learning

### Before Starting
1. What specific task do I want to solve?
2. How will I measure success?
3. Do I have enough quality data?
4. What compute resources do I have?
5. How will I know if finetuning helped?

### During Training
1. Is my loss decreasing?
2. Is validation loss following training loss?
3. Am I overfitting?
4. Do I need to adjust hyperparameters?
5. How do my metrics compare to baseline?

### After Training
1. Did finetuning improve over baseline?
2. Does the model generalize to new data?
3. What errors is the model making?
4. Can I improve with more data or better hyperparameters?
5. Is the model ready for production?

---

## Recommended Resources for Deep Diving

### Conceptual Understanding
- **Attention Is All You Need** (Transformer paper)
- **BERT paper** (Bidirectional understanding)
- **GPT papers** (Autoregressive generation)
- **LoRA paper** (Efficient finetuning)

### Practical Implementation
- **Hugging Face Documentation** (Best library for finetuning)
- **PEFT Documentation** (LoRA, QLoRA implementation)
- **DeepSpeed/FSDP** (Scaling to larger models)

### Blogs & Tutorials
- **Hugging Face Blog** (Practical guides)
- **Sebastian Ruder's Blog** (Transfer learning)
- **Jay Alammar's Blog** (Visual explanations)

### Communities
- **Hugging Face Forums**
- **r/MachineLearning**
- **Papers With Code**

---

## Final Thoughts

### Key Takeaways

**1. Start Simple**
- Use LoRA first
- Small model first
- Small dataset first
- Get something working, then improve

**2. Quality > Quantity**
- 500 good examples > 5000 bad examples
- Clean data is crucial
- Understand your data before training

**3. Iterate**
- Finetuning is experimental
- Try different hyperparameters
- Learn from failures
- Track everything with MLflow

**4. Measure Properly**
- Always have a baseline
- Use appropriate metrics
- Do qualitative analysis
- Real-world testing matters

**5. Be Patient**
- First attempts often fail
- Learning takes time
- Each experiment teaches you something
- Success comes from iteration

### You're Ready!

You now understand:
- ✓ What finetuning is and why it works
- ✓ Different finetuning methods (LoRA, QLoRA, etc.)
- ✓ The complete workflow from data to evaluation
- ✓ How to integrate with DVC and MLflow
- ✓ Common pitfalls and how to avoid them
- ✓ How to make practical decisions

**Next step:** Get your hands dirty with code!

Pick a simple task, prepare some data, and start experimenting. The concepts in this guide will make much more sense once you've trained your first model.

Good luck with your finetuning journey!