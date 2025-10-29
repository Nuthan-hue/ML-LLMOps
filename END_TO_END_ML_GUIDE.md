# End-to-End Data Science & ML Workflows Guide

> A comprehensive guide covering the complete ML lifecycle from data collection to deployment, with multiple approaches and tools.

## Table of Contents

1. [The Complete ML Lifecycle](#the-complete-ml-lifecycle)
2. [Project Types & Approaches](#project-types--approaches)
3. [Phase 1: Data Collection](#phase-1-data-collection)
4. [Phase 2: Exploratory Data Analysis (EDA)](#phase-2-exploratory-data-analysis-eda)
5. [Phase 3: Data Preprocessing](#phase-3-data-preprocessing)
6. [Phase 4: Feature Engineering](#phase-4-feature-engineering)
7. [Phase 5: Model Development](#phase-5-model-development)
8. [Phase 6: Model Evaluation](#phase-6-model-evaluation)
9. [Phase 7: Experiment Tracking](#phase-7-experiment-tracking)
10. [Phase 8: Model Deployment](#phase-8-model-deployment)
11. [Phase 9: Monitoring & Maintenance](#phase-9-monitoring--maintenance)
12. [MLOps Tools Ecosystem](#mlops-tools-ecosystem)
13. [Project Templates](#project-templates)

---

## The Complete ML Lifecycle

```
Data Collection → EDA → Data Preprocessing → Feature Engineering
        ↓
    Modeling → Evaluation → Experiment Tracking
        ↓
  Deployment → Monitoring → Retraining
        ↓
   (Continuous Loop)
```

### Key Principles

1. **Iterative Process**: ML is not linear; you'll loop back often
2. **Version Everything**: Code, data, models, experiments
3. **Reproducibility First**: Anyone should be able to reproduce your results
4. **Automation**: Automate repetitive tasks with pipelines
5. **Documentation**: Document decisions, experiments, and learnings

---

## Project Types & Approaches

### 1. Traditional ML (Tabular Data)
**Examples**: Customer churn, fraud detection, price prediction

**Typical Stack**:
- **Data**: Pandas, NumPy
- **Modeling**: Scikit-learn, XGBoost, LightGBM
- **Evaluation**: Scikit-learn metrics

**Your Current Example**: `ElasticNetWineModel_Dagshub.py`

---

### 2. Deep Learning (Images, Time Series, NLP)
**Examples**: Image classification, object detection, sentiment analysis

**Typical Stack**:
- **Data**: PyTorch DataLoader, TensorFlow Dataset
- **Modeling**: PyTorch, TensorFlow, Keras
- **Preprocessing**: torchvision, PIL, OpenCV

---

### 3. LLM & NLP
**Examples**: Text generation, chatbots, fine-tuning, RAG

**Typical Stack**:
- **Data**: Hugging Face Datasets
- **Modeling**: Transformers, PEFT (LoRA/QLoRA)
- **Inference**: Ollama, vLLM, TGI (Text Generation Inference)

**Your Guide**: `FINETUNING_GUIDE.md`

---

## Phase 1: Data Collection

### Approach 1: Public Datasets

#### Method 1.1: Direct Download
```python
import pandas as pd

# From URL
df = pd.read_csv('https://example.com/dataset.csv')

# From local file
df = pd.read_csv('data/raw/dataset.csv')
```

#### Method 1.2: Hugging Face Datasets
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")  # Movie reviews
dataset = load_dataset("squad")  # Question answering
dataset = load_dataset("mnist")  # Images

# Convert to pandas
df = dataset['train'].to_pandas()
```

#### Method 1.3: Kaggle API
```bash
# Install
pip install kaggle

# Setup credentials (~/.kaggle/kaggle.json)
kaggle datasets download -d dataset-name

# In Python
import kaggle
kaggle.api.dataset_download_files('dataset-name', path='data/raw/', unzip=True)
```

#### Method 1.4: OpenML
```python
from sklearn.datasets import fetch_openml

# Fetch dataset
data = fetch_openml('wine-quality-red', version=1, as_frame=True)
df = data.frame
```

---

### Approach 2: APIs & Web Scraping

#### Method 2.1: REST APIs
```python
import requests
import pandas as pd

# Example: Weather API
response = requests.get('https://api.weather.com/data',
                       params={'key': 'YOUR_API_KEY'})
data = response.json()
df = pd.DataFrame(data)
```

#### Method 2.2: Web Scraping
```python
from bs4 import BeautifulSoup
import requests

url = 'https://example.com/data'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract data
data = []
for item in soup.find_all('div', class_='data-item'):
    data.append({
        'title': item.find('h2').text,
        'value': item.find('span', class_='value').text
    })

df = pd.DataFrame(data)
```

#### Method 2.3: Twitter/Social Media APIs
```python
import tweepy

# Setup
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

# Collect tweets
tweets = api.search_tweets(q='machine learning', count=100)
df = pd.DataFrame([tweet.text for tweet in tweets], columns=['text'])
```

---

### Approach 3: Databases

#### Method 3.1: SQL Databases
```python
import pandas as pd
import sqlite3

# SQLite
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

# PostgreSQL
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost/dbname')
df = pd.read_sql_query("SELECT * FROM customers", engine)
```

#### Method 3.2: NoSQL (MongoDB)
```python
from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
collection = db['customers']

# Query and convert to DataFrame
data = list(collection.find({}))
df = pd.DataFrame(data)
```

---

### Data Versioning with DVC

Once you have data, version it:

```bash
# Initialize DVC
dvc init

# Add data to DVC tracking
dvc add data/raw/dataset.csv

# Commit .dvc file to git
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw dataset"

# Push to remote storage
dvc remote add -d storage s3://mybucket/dvc-storage
dvc push
```

**Your current setup**: Already has DVC initialized (`.dvc/` directory)

---

## Phase 2: Exploratory Data Analysis (EDA)

> **Goal**: Understand your data before modeling

### Approach 1: Basic Pandas EDA

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/dataset.csv')

# ========================================
# 1. OVERVIEW
# ========================================
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.columns)

# ========================================
# 2. MISSING VALUES
# ========================================
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing'] > 0])

# ========================================
# 3. DUPLICATES
# ========================================
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# ========================================
# 4. DATA TYPES
# ========================================
print(df.dtypes)

# ========================================
# 5. UNIQUE VALUES
# ========================================
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# ========================================
# 6. VALUE COUNTS (Categorical)
# ========================================
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ========================================
# 7. DISTRIBUTIONS (Numerical)
# ========================================
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

# ========================================
# 8. OUTLIERS (IQR Method)
# ========================================
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")

# ========================================
# 9. CORRELATIONS
# ========================================
correlation_matrix = df.corr()
print(correlation_matrix)
```

---

### Approach 2: Visualization-Based EDA

#### Method 2.1: Matplotlib & Seaborn
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ========================================
# 1. DISTRIBUTIONS
# ========================================
# Histograms for all numerical columns
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.savefig('reports/figures/distributions.png')
plt.close()

# Individual distribution with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['column_name'], kde=True, bins=30)
plt.title('Distribution of Column Name')
plt.savefig('reports/figures/column_dist.png')
plt.close()

# ========================================
# 2. BOX PLOTS (Outliers)
# ========================================
plt.figure(figsize=(15, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.title('Box Plots - Outlier Detection')
plt.tight_layout()
plt.savefig('reports/figures/boxplots.png')
plt.close()

# ========================================
# 3. CORRELATION HEATMAP
# ========================================
plt.figure(figsize=(12, 10))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png')
plt.close()

# ========================================
# 4. PAIRPLOT (Relationships)
# ========================================
# Warning: Can be slow with many columns
sns.pairplot(df, hue='target_column')
plt.savefig('reports/figures/pairplot.png')
plt.close()

# ========================================
# 5. COUNT PLOTS (Categorical)
# ========================================
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='category_column', order=df['category_column'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribution of Categories')
plt.tight_layout()
plt.savefig('reports/figures/category_dist.png')
plt.close()

# ========================================
# 6. VIOLIN PLOTS
# ========================================
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='category_column', y='numerical_column')
plt.title('Violin Plot')
plt.tight_layout()
plt.savefig('reports/figures/violin_plot.png')
plt.close()

# ========================================
# 7. TIME SERIES (if applicable)
# ========================================
plt.figure(figsize=(15, 6))
df['date'] = pd.to_datetime(df['date'])
df.set_index('date')['value'].plot()
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('reports/figures/timeseries.png')
plt.close()
```

#### Method 2.2: Plotly (Interactive)
```python
import plotly.express as px
import plotly.graph_objects as go

# Scatter plot
fig = px.scatter(df, x='feature1', y='feature2', color='target',
                 title='Interactive Scatter Plot')
fig.write_html('reports/figures/interactive_scatter.html')

# 3D Scatter
fig = px.scatter_3d(df, x='f1', y='f2', z='f3', color='target')
fig.write_html('reports/figures/3d_scatter.html')

# Correlation heatmap
fig = px.imshow(df.corr(), text_auto=True, aspect="auto")
fig.write_html('reports/figures/correlation_interactive.html')
```

---

### Approach 3: Automated EDA Tools

#### Method 3.1: Pandas Profiling (ydata-profiling)
```python
from ydata_profiling import ProfileReport

# Generate comprehensive report
profile = ProfileReport(df,
                       title="Dataset EDA Report",
                       explorative=True,
                       dark_mode=False)

# Save report
profile.to_file("reports/eda_report.html")
```

**What it includes**:
- Overview (dataset size, types, missing values)
- Variables (distributions, statistics, outliers)
- Correlations (Pearson, Spearman, Kendall)
- Missing values patterns
- Duplicate rows
- Interactions between variables

#### Method 3.2: Sweetviz
```python
import sweetviz as sv

# Generate report
report = sv.analyze(df)
report.show_html('reports/sweetviz_report.html')

# Compare train vs test
report = sv.compare([train, "Train"], [test, "Test"])
report.show_html('reports/train_vs_test.html')

# Compare with target
report = sv.compare_intra(df, df["target"] == 1, ["Target=1", "Target=0"])
report.show_html('reports/target_comparison.html')
```

#### Method 3.3: AutoViz
```python
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
dft = AV.AutoViz(
    filename='data/raw/dataset.csv',
    depVar='target_column',
    dfte=None,
    header=0,
    verbose=1,
    lowess=False,
    chart_format='png',
    max_rows_analyzed=150000,
    max_cols_analyzed=30
)
```

#### Method 3.4: D-Tale
```python
import dtale

# Launch interactive dashboard
d = dtale.show(df)
d.open_browser()  # Opens in browser with full interactivity

# Can also export
d.main_url()  # Get URL
```

---

### Approach 4: Statistical Analysis

```python
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========================================
# 1. NORMALITY TESTS
# ========================================
for col in df.select_dtypes(include=[np.number]).columns:
    stat, p_value = stats.shapiro(df[col].dropna())
    print(f"{col}: p-value = {p_value:.4f}")
    if p_value > 0.05:
        print(f"  → Appears normally distributed")
    else:
        print(f"  → Not normally distributed")

# ========================================
# 2. SKEWNESS & KURTOSIS
# ========================================
print("\nSkewness:")
print(df.skew())
print("\nKurtosis:")
print(df.kurtosis())

# ========================================
# 3. HYPOTHESIS TESTING
# ========================================
# T-test for two groups
group1 = df[df['category'] == 'A']['value']
group2 = df[df['category'] == 'B']['value']
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")

# Chi-square test for independence
contingency_table = pd.crosstab(df['cat1'], df['cat2'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square: χ²={chi2:.4f}, p={p_value:.4f}")

# ========================================
# 4. MULTICOLLINEARITY (VIF)
# ========================================
numerical_cols = df.select_dtypes(include=[np.number]).columns
vif_data = pd.DataFrame()
vif_data["Feature"] = numerical_cols
vif_data["VIF"] = [variance_inflation_factor(df[numerical_cols].values, i)
                   for i in range(len(numerical_cols))]
print("\nVariance Inflation Factor:")
print(vif_data)
# VIF > 10 indicates multicollinearity
```

---

### EDA Best Practices

1. **Always start with basic stats**: `df.info()`, `df.describe()`
2. **Check data quality first**: Missing values, duplicates, data types
3. **Visualize distributions**: Understand your data shape
4. **Look for relationships**: Correlations, dependencies
5. **Identify outliers**: Decide how to handle them
6. **Document findings**: Save plots, write observations
7. **Generate automated reports**: Use pandas-profiling for quick insights

---

## Phase 3: Data Preprocessing

### Approach 1: Handling Missing Values

#### Method 1.1: Deletion
```python
# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows where specific column is missing
df_clean = df.dropna(subset=['important_column'])

# Drop columns with too many missing values
threshold = 0.5  # 50%
df_clean = df.dropna(thresh=len(df) * threshold, axis=1)
```

#### Method 1.2: Imputation - Simple
```python
from sklearn.impute import SimpleImputer

# Mean/Median imputation for numerical
imputer = SimpleImputer(strategy='mean')  # or 'median'
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Mode imputation for categorical
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

# Constant value
imputer = SimpleImputer(strategy='constant', fill_value=0)
df[cols] = imputer.fit_transform(df[cols])
```

#### Method 1.3: Imputation - Advanced
```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df[numerical_cols])

# Iterative (MICE - Multiple Imputation)
imputer = IterativeImputer(random_state=42, max_iter=10)
df_imputed = imputer.fit_transform(df[numerical_cols])
```

---

### Approach 2: Handling Outliers

#### Method 2.1: Removal (IQR Method)
```python
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) &
                           (df_clean[col] <= upper_bound)]
    return df_clean

df_clean = remove_outliers_iqr(df, numerical_columns)
```

#### Method 2.2: Capping (Winsorization)
```python
from scipy.stats.mstats import winsorize

# Cap at 1st and 99th percentile
df['column'] = winsorize(df['column'], limits=[0.01, 0.01])

# Manual capping
lower = df['column'].quantile(0.01)
upper = df['column'].quantile(0.99)
df['column'] = df['column'].clip(lower, upper)
```

#### Method 2.3: Transformation
```python
import numpy as np

# Log transformation (for right-skewed data)
df['column_log'] = np.log1p(df['column'])

# Square root transformation
df['column_sqrt'] = np.sqrt(df['column'])

# Box-Cox transformation
from scipy.stats import boxcox
df['column_boxcox'], _ = boxcox(df['column'] + 1)  # +1 if data has zeros
```

---

### Approach 3: Encoding Categorical Variables

#### Method 3.1: Label Encoding (Ordinal)
```python
from sklearn.preprocessing import LabelEncoder

# For ordinal data (low, medium, high)
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# Or manually with mapping
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_map)
```

#### Method 3.2: One-Hot Encoding (Nominal)
```python
# Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['category1', 'category2'],
                            drop_first=True)  # Avoid dummy variable trap

# Scikit-learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded = ohe.fit_transform(df[['category']])
encoded_df = pd.DataFrame(encoded,
                         columns=ohe.get_feature_names_out(['category']))
df_encoded = pd.concat([df, encoded_df], axis=1)
```

#### Method 3.3: Target Encoding (For high cardinality)
```python
# Mean target encoding
category_means = df.groupby('high_cardinality_col')['target'].mean()
df['category_encoded'] = df['high_cardinality_col'].map(category_means)

# Using category_encoders library
from category_encoders import TargetEncoder

te = TargetEncoder()
df['category_encoded'] = te.fit_transform(df['high_cardinality_col'],
                                          df['target'])
```

#### Method 3.4: Frequency Encoding
```python
# Replace category with its frequency
freq = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq)
```

---

### Approach 4: Scaling & Normalization

#### Method 4.1: Standardization (Z-score)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Result: mean=0, std=1
```

**When to use**: Most ML algorithms (SVM, Neural Networks, PCA)

#### Method 4.2: Min-Max Scaling
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Result: values between 0 and 1
```

**When to use**: Neural networks, image processing, bounded ranges needed

#### Method 4.3: Robust Scaling
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Uses median and IQR - robust to outliers
```

**When to use**: Data with outliers

#### Method 4.4: MaxAbs Scaling
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Result: values between -1 and 1
```

**When to use**: Sparse data

---

### Approach 5: Text Preprocessing (For NLP)

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # 4. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Remove extra whitespace
    text = ' '.join(text.split())

    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatization (or Stemming)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)
```

---

### Pipeline Approach (Best Practice)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column types
numerical_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city', 'category']

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Use in model
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

---

## Phase 4: Feature Engineering

### Approach 1: Creating New Features

#### Domain-Specific Features
```python
# Example: E-commerce
df['revenue'] = df['quantity'] * df['price']
df['discount_rate'] = (df['original_price'] - df['sale_price']) / df['original_price']
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Example: Finance
df['debt_to_income'] = df['debt'] / df['income']
df['credit_utilization'] = df['credit_used'] / df['credit_limit']

# Example: Time-based
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_holiday'] = df['date'].isin(holidays).astype(int)
```

---

### Approach 2: Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create interaction and polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])

# Get feature names
feature_names = poly.get_feature_names_out(['feature1', 'feature2'])
df_poly = pd.DataFrame(poly_features, columns=feature_names)
```

---

### Approach 3: Binning/Discretization

```python
# Equal-width binning
df['age_bin'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Old'])

# Custom bins
df['income_category'] = pd.cut(df['income'],
                               bins=[0, 30000, 60000, 100000, float('inf')],
                               labels=['Low', 'Medium', 'High', 'Very High'])

# Quantile-based binning
df['score_quartile'] = pd.qcut(df['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

---

### Approach 4: Feature Selection

#### Method 4.1: Correlation-based
```python
# Remove highly correlated features
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns
           if any(upper_triangle[column] > 0.95)]
df_reduced = df.drop(columns=to_drop)
```

#### Method 4.2: Statistical Tests
```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# For classification
selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
```

#### Method 4.3: Model-based (Feature Importance)
```python
from sklearn.ensemble import RandomForestClassifier

# Train a model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top N features
top_features = feature_importance.head(20)['feature'].tolist()
X_train_selected = X_train[top_features]
```

#### Method 4.4: Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# Selected features
selected_features = X_train.columns[selector.support_].tolist()
```

---

## Phase 5: Model Development

### Approach 1: Traditional ML (Scikit-learn)

#### Classification

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================
# Model 1: Logistic Regression
# ========================================
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# ========================================
# Model 2: Random Forest
# ========================================
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)

# ========================================
# Model 3: XGBoost
# ========================================
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# ========================================
# Model 4: LightGBM
# ========================================
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
lgb_clf.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

#### Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import xgboost as xgb

# ========================================
# Model 1: Linear Regression with Regularization
# ========================================
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

ridge.fit(X_train, y_train)

# ========================================
# Model 2: Random Forest Regressor
# ========================================
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# ========================================
# Model 3: XGBoost Regressor
# ========================================
xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_reg.fit(X_train, y_train)
```

---

### Approach 2: Deep Learning (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========================================
# 1. DEFINE DATASET
# ========================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ========================================
# 2. DEFINE MODEL
# ========================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=128, output_size=1)

# ========================================
# 3. TRAINING
# ========================================
criterion = nn.BCEWithLogitsLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ========================================
# 4. EVALUATION
# ========================================
model.eval()
with torch.no_grad():
    test_outputs = model(torch.FloatTensor(X_test))
    predictions = (torch.sigmoid(test_outputs) > 0.5).float()
```

---

### Approach 3: Hyperparameter Tuning

#### Method 3.1: Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_model = grid_search.best_estimator_
```

#### Method 3.2: Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

#### Method 3.3: Optuna (Bayesian Optimization)
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # Cross-validation score
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)

# Train final model
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
```

---

## Phase 6: Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ========================================
# 1. BASIC METRICS
# ========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ========================================
# 2. CLASSIFICATION REPORT
# ========================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ========================================
# 3. CONFUSION MATRIX
# ========================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('reports/figures/confusion_matrix.png')
plt.close()

# ========================================
# 4. ROC-AUC
# ========================================
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('reports/figures/roc_curve.png')
plt.close()

# ========================================
# 5. PRECISION-RECALL CURVE
# ========================================
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('reports/figures/pr_curve.png')
plt.close()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

y_pred = model.predict(X_test)

# ========================================
# METRICS
# ========================================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.4f}")

# ========================================
# RESIDUAL PLOT
# ========================================
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('reports/figures/residual_plot.png')
plt.close()

# ========================================
# ACTUAL VS PREDICTED
# ========================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.savefig('reports/figures/actual_vs_predicted.png')
plt.close()
```

---

## Phase 7: Experiment Tracking

### Approach 1: MLflow

#### Basic Logging

```python
import mlflow
import mlflow.sklearn

# Set experiment
mlflow.set_experiment("wine-quality-prediction")

# Start run
with mlflow.start_run(run_name="random-forest-v1"):

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("train_size", len(X_train))

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("reports/figures/confusion_matrix.png")

    # Log dataset
    mlflow.log_artifact("data/processed/train.csv")
```

#### Advanced: Autologging

```python
import mlflow.sklearn

# Enable autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # Automatically logs parameters, metrics, and model!
```

#### DagsHub Integration (Your Current Setup)

```python
import dagshub
import mlflow

# Initialize DagsHub
dagshub.init(repo_owner='nuthan.maddineni23',
             repo_name='ML-LLMOps',
             mlflow=True)

# Set tracking URI
mlflow.set_tracking_uri('https://dagshub.com/nuthan.maddineni23/ML-LLMOps.mlflow')

# Now use MLflow as normal
with mlflow.start_run():
    # Your training code
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    mlflow.log_model(model, "model")
```

---

### Approach 2: Weights & Biases (wandb)

```python
import wandb

# Initialize
wandb.init(
    project="wine-quality",
    name="random-forest-v1",
    config={
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1
    }
)

# Log metrics during training
for epoch in range(num_epochs):
    # Training
    train_loss = train_one_epoch()
    val_loss = validate()

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

# Log final metrics
wandb.log({
    "final_accuracy": accuracy,
    "final_f1": f1_score
})

# Log model
wandb.log_artifact("model.pkl")

# Log plots
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

# Finish run
wandb.finish()
```

---

### Approach 3: TensorBoard (For Deep Learning)

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log scalars
for epoch in range(num_epochs):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

# Log model graph
writer.add_graph(model, input_sample)

# Log images
writer.add_image('predictions', img_grid, epoch)

# Log hyperparameters
writer.add_hparams(
    {'lr': 0.001, 'batch_size': 32},
    {'accuracy': final_accuracy, 'loss': final_loss}
)

writer.close()

# View with: tensorboard --logdir=runs
```

---

## Phase 8: Model Deployment

### Approach 1: Flask API (Simple)

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Preprocess
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t ml-api .
docker run -p 5000:5000 ml-api

# Test
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

---

### Approach 2: FastAPI (Modern, with docs)

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model
model = joblib.load('models/model.pkl')

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return PredictionOutput(
            prediction=int(prediction[0]),
            probability=float(probability[0][1])
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run with: uvicorn app:app --reload
# Auto docs at: http://localhost:8000/docs
```

---

### Approach 3: Streamlit (Interactive Dashboard)

```python
# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/model.pkl')

model = load_model()

# Title
st.title("ML Model Prediction Dashboard")

# Sidebar
st.sidebar.header("Input Features")

# Input features
feature1 = st.sidebar.slider("Feature 1", 0.0, 10.0, 5.0)
feature2 = st.sidebar.slider("Feature 2", 0.0, 10.0, 5.0)
feature3 = st.sidebar.slider("Feature 3", 0.0, 10.0, 5.0)

# Predict button
if st.sidebar.button("Predict"):
    # Create input
    input_data = np.array([[feature1, feature2, feature3]])

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Prediction: {prediction[0]}")
    st.write(f"Probability: {probability[0][1]:.4f}")

    # Visualize
    st.bar_chart(pd.DataFrame({
        'Class 0': [probability[0][0]],
        'Class 1': [probability[0][1]]
    }))

# Run with: streamlit run app.py
```

---

### Approach 4: Docker Compose with Ollama (Your Setup)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ML API
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pkl
    depends_on:
      - ollama

  # Ollama for LLM inference
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: serve

  # Optional: MLflow server
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow

volumes:
  ollama_data:
```

---

### Approach 5: Cloud Deployment

#### AWS SageMaker
```python
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Create model
sklearn_model = SKLearnModel(
    model_data='s3://bucket/model.tar.gz',
    role='arn:aws:iam::123456789:role/SageMakerRole',
    entry_point='inference.py',
    framework_version='1.0-1'
)

# Deploy
predictor = sklearn_model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)

# Predict
prediction = predictor.predict(data)
```

#### Google Cloud AI Platform
```bash
# Upload model
gsutil cp model.pkl gs://bucket/model/

# Create model version
gcloud ai-platform versions create v1 \
  --model=my_model \
  --origin=gs://bucket/model/ \
  --runtime-version=2.8 \
  --framework=SCIKIT_LEARN

# Predict
gcloud ai-platform predict \
  --model=my_model \
  --version=v1 \
  --json-instances=instances.json
```

---

## Phase 9: Monitoring & Maintenance

### Approach 1: Model Monitoring

```python
# monitoring.py
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self, reference_data, model):
        self.reference_data = reference_data
        self.model = model
        self.reference_predictions = model.predict(reference_data)

    def detect_data_drift(self, new_data, threshold=0.05):
        """
        Detect data drift using Kolmogorov-Smirnov test
        """
        drift_detected = {}

        for column in self.reference_data.columns:
            # KS test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[column],
                new_data[column]
            )

            drift_detected[column] = {
                'drift': p_value < threshold,
                'p_value': p_value,
                'statistic': statistic
            }

        return drift_detected

    def detect_prediction_drift(self, new_predictions, threshold=0.1):
        """
        Detect if prediction distribution has changed
        """
        ref_mean = np.mean(self.reference_predictions)
        ref_std = np.std(self.reference_predictions)

        new_mean = np.mean(new_predictions)
        new_std = np.std(new_predictions)

        mean_drift = abs(ref_mean - new_mean) / ref_std > threshold

        return {
            'drift_detected': mean_drift,
            'reference_mean': ref_mean,
            'new_mean': new_mean,
            'difference': abs(ref_mean - new_mean)
        }

    def calculate_performance_metrics(self, X, y_true):
        """
        Calculate current performance metrics
        """
        y_pred = self.model.predict(X)

        from sklearn.metrics import accuracy_score, f1_score

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'timestamp': datetime.now().isoformat()
        }

# Usage
monitor = ModelMonitor(X_train, model)

# Check for drift in new data
drift_results = monitor.detect_data_drift(X_new)
print("Drift detected:", drift_results)

# Check prediction drift
pred_drift = monitor.detect_prediction_drift(model.predict(X_new))
if pred_drift['drift_detected']:
    print("Alert: Prediction drift detected!")
```

---

### Approach 2: Logging & Alerts

```python
# logging_config.py
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/model_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# Log predictions
def log_prediction(input_data, prediction, probability, latency):
    logger.info(f"Prediction made: {prediction}, Probability: {probability:.4f}, Latency: {latency:.3f}ms")

# Send alerts
def send_alert(message, alert_type='warning'):
    if alert_type == 'critical':
        # Send email, Slack, PagerDuty, etc.
        logger.critical(message)
    else:
        logger.warning(message)
```

---

## MLOps Tools Ecosystem

### Complete Toolchain

| Stage | Tool Options | Your Project |
|-------|--------------|--------------|
| **Version Control** | Git, GitHub, GitLab | ✅ Git |
| **Data Versioning** | DVC, LakeFS, Delta Lake | ✅ DVC |
| **Experiment Tracking** | MLflow, wandb, Neptune | ✅ MLflow + DagsHub |
| **Feature Store** | Feast, Tecton, Hopsworks | ⚪ Add if needed |
| **Model Registry** | MLflow, wandb | ✅ MLflow |
| **Pipeline Orchestration** | DVC, Airflow, Prefect, Kubeflow | ✅ DVC |
| **Containerization** | Docker, Podman | ✅ Docker |
| **Deployment** | Flask, FastAPI, Streamlit | ⚪ To add |
| **Model Serving** | Ollama, TorchServe, TF Serving | ✅ Ollama |
| **Monitoring** | Prometheus, Grafana, Evidently | ⚪ To add |
| **CI/CD** | GitHub Actions, GitLab CI, Jenkins | ⚪ To add |

---

## Project Templates

### Template 1: Traditional ML Project

```
project/
├── data/
│   ├── raw/                 # Original data
│   ├── processed/           # Cleaned data
│   └── features/            # Feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       ├── helpers.py
│       └── visualization.py
├── models/                  # Saved models
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
│   └── test_*.py
├── .dvc/
├── .gitignore
├── dvc.yaml                # DVC pipeline
├── params.yaml             # Parameters
├── requirements.txt
└── README.md
```

---

### Template 2: LLM Fine-tuning Project

```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── finetuning/         # Formatted for training
├── configs/
│   ├── lora_config.yaml
│   └── training_config.yaml
├── src/
│   ├── data/
│   │   └── prepare_finetuning_data.py
│   ├── models/
│   │   ├── load_base_model.py
│   │   ├── finetune.py
│   │   └── evaluate.py
│   └── inference/
│       ├── generate.py
│       └── api.py
├── models/
│   ├── base/               # Base model cache
│   └── finetuned/          # Your fine-tuned models
├── notebooks/
│   └── experiments/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .dvc/
├── dvc.yaml
├── requirements.txt
├── FINETUNING_GUIDE.md
└── README.md
```

---

### Template 3: Full MLOps Project (Recommended)

```
project/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
├── notebooks/
│   ├── eda/
│   └── experiments/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── schemas.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detection.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── models/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   └── metrics/
├── configs/
│   ├── config.yaml
│   └── params.yaml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── .dvc/
├── .dvcignore
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── Makefile
├── README.md
└── LICENSE
```

---

## Quick Start Commands

```bash
# Project setup
mkdir my-ml-project && cd my-ml-project
git init
dvc init
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Data workflow
python src/data/data_ingestion.py
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw data"

# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Run preprocessing
python src/data/data_preprocessing.py
dvc add data/processed/

# Train model
python src/models/train.py

# Evaluate
python src/models/evaluate.py

# DVC pipeline
dvc repro

# MLflow UI
mlflow ui

# Deploy
docker-compose up

# Monitor
python src/monitoring/drift_detection.py
```

---

## Best Practices Summary

1. **Version everything**: Code (Git), Data (DVC), Models (MLflow)
2. **Experiment systematically**: Track all experiments with MLflow
3. **Automate with pipelines**: Use DVC pipelines for reproducibility
4. **Test your code**: Write tests for data, features, and models
5. **Document thoroughly**: README, docstrings, notebooks
6. **Monitor in production**: Track performance, drift, and errors
7. **Iterate continuously**: ML is an iterative process

---

## Next Steps

1. Choose a project type that matches your goal
2. Set up the project structure
3. Follow the phases sequentially
4. Track everything with MLflow + DVC
5. Deploy when ready
6. Monitor and improve

Remember: Start simple, iterate, and gradually add complexity!