# # 🫀 Heart Failure Prediction with PySpark 

A personal Machine Learning project that predicts the risk of death among heart failure patients based on clinical data.

The project uses **PySpark and Spark MLlib** to preprocess clinical data and train a **Logistic Regression** model. The trained model is then integrated into a **Tkinter desktop application**, allowing users to enter patient information and receive a prediction through a simple graphical interface.

The project also includes additional **KMeans clustering** and **FP-Growth association rule mining** analysis to explore patterns within the dataset.

---

## ✨ Highlights

- Heart failure risk prediction using Logistic Regression
- Interactive desktop application built with Tkinter
- Machine Learning pipeline with PySpark / Spark MLlib
- Data processing with HDFS
- Model training and evaluation
- KMeans clustering for patient segmentation
- FP-Growth for association rule mining
---

## ## 🏗️ Architecture

The project is organized around a machine learning prediction pipeline, with the Tkinter application serving as the user-facing layer.

```text
                         ┌──────────────────────┐
                         │         HDFS         │
                         │  Heart Failure CSV   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      PySpark         │
                         │   Data Processing    │
                         │  • Cleaning          │
                         │  • Feature Processing│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Logistic Regression │
                         │    Model Training    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Trained Model     │
                         │  Spark ML Pipeline   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Tkinter GUI      │
                         │                      │
                         │  Patient Information │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Prediction Result  │
                         │   Death Event Risk   │
                         └──────────────────────┘
```

### Additional Data Mining Analysis

In addition to the prediction pipeline, the project includes two exploratory data mining techniques:

```text
                  Clinical Dataset
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          KMeans                FP-Growth
       Clustering          Association Rules
              │                     │
              ▼                     ▼
      Patient Profiles       Frequent Patterns
```

- **KMeans** is used to identify groups of patients with similar clinical characteristics.
- **FP-Growth** is used to discover frequent patterns and association rules among discretized clinical features.

---

## 📊 Dataset

**Dataset:** `heart_failure_clinical_records_dataset.csv`

The dataset contains clinical information from patients with heart failure.

The target variable for the classification task is:

```text
DEATH_EVENT
```

where the model learns to predict whether a death event occurred during the follow-up period.

The dataset is loaded from **HDFS** to demonstrate a distributed data processing workflow with Spark.

---

## 🧠 Methods

### 1. Logistic Regression

Used as a binary classification model to predict `DEATH_EVENT`.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- PR AUC

Prediction outputs are stored in:

```text
outputs/classification_metrics/
outputs/classification_predictions/
```

---

### 2. KMeans

KMeans is used to discover groups of patients with similar clinical characteristics.

Different values of `k` are evaluated using **Silhouette Score** to select an appropriate number of clusters.

The resulting clusters are further analyzed using:

- Silhouette Score
- Cluster Purity
- Cluster size
- Mortality rate
- Average clinical features

Outputs:

```text
outputs/kmeans_k_selection/
outputs/kmeans_cluster_metrics/
outputs/kmeans_cluster_profile/
```

---

### 3. FP-Growth

FP-Growth is applied to discover frequent patterns and association rules from discretized clinical features.

The generated rules are evaluated using:

- Support
- Confidence
- Lift
- All-confidence
- Kulczynski
- Cosine

Outputs:

```text
outputs/fpgrowth_itemsets/
outputs/fpgrowth_rules/
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Big Data | Apache Spark / PySpark |
| Distributed Storage | HDFS |
| Machine Learning | Spark MLlib |
| Classification | Logistic Regression |
| Clustering | KMeans |
| Pattern Mining | FP-Growth |
| GUI | Tkinter |
| Version Control | Git / GitHub |
| Runtime | Java |

---

## 🚀 Getting Started

### Prerequisites

Make sure the following are installed:

- Python 3
- Java
- Apache Spark
- Hadoop / HDFS
- Git

Verify Java:

```bash
java -version
```

Verify Python:

```bash
python3 --version
```

---

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Data Mining Pipeline

Run the main PySpark pipeline:

```bash
python src/heart_failure_pyspark.py \
  --input <YOUR_HDFS_PATH>/heart_failure_clinical_records_dataset.csv \
  --output outputs
```

Example:

```bash
python src/heart_failure_pyspark.py \
  --input hdfs://localhost:9000/data/heart_failure_clinical_records_dataset.csv \
  --output outputs
```

After execution, the analysis results will be available under:

```text
outputs/
```

The summary report can be found at:

```text
outputs/comparison_summary.md
```

---

## 🤖 Train the Prediction Model

A dedicated training script is provided to create a reusable Logistic Regression pipeline.

```bash
python src/train_heart_failure_model.py
```

Or specify the input and model output paths explicitly:

```bash
python src/train_heart_failure_model.py \
  --input <YOUR_HDFS_PATH>/heart_failure_clinical_records_dataset.csv \
  --model-output models/logistic_regression_pipeline
```

The trained pipeline is saved to:

```text
models/logistic_regression_pipeline/
```

---

## 🖥️ Run the Prediction Application

After training the model, launch the Tkinter application:

```bash
python src/predict_heart_failure_tkinter.py
```

The application allows users to enter patient clinical features and obtain a prediction from the trained Logistic Regression pipeline.

```text
Patient Features
       │
       ▼
   Tkinter GUI
       │
       ▼
Trained ML Pipeline
       │
       ▼
 Prediction
```

> **Note:** This application is intended for demonstration and educational purposes only.

---

## 📂 Project Structure

```text
.
├── src/
│   ├── heart_failure_pyspark.py
│   ├── train_heart_failure_model.py
│   └── predict_heart_failure_tkinter.py
│
├── models/
│   └── logistic_regression_pipeline/
│
├── outputs/
│   ├── classification_metrics/
│   ├── classification_predictions/
│   ├── kmeans_k_selection/
│   ├── kmeans_cluster_metrics/
│   ├── kmeans_cluster_profile/
│   ├── fpgrowth_itemsets/
│   ├── fpgrowth_rules/
│   └── comparison_summary.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Outputs

The pipeline generates the following artifacts:

| Output | Description |
|---|---|
| `classification_metrics/` | Classification evaluation metrics |
| `classification_predictions/` | Logistic Regression predictions |
| `kmeans_k_selection/` | Silhouette scores for candidate `k` values |
| `kmeans_cluster_metrics/` | Metrics for the selected clustering model |
| `kmeans_cluster_profile/` | Statistical profile of each cluster |
| `fpgrowth_itemsets/` | Frequent itemsets |
| `fpgrowth_rules/` | Association rules and rule metrics |
| `comparison_summary.md` | Summary comparing the three approaches |

---

## 🔍 Key Questions

This project is designed to explore three main questions:

**1. Classification**

> Can clinical features be used to predict whether a patient experiences a death event?

**2. Clustering**

> Can patients be grouped into meaningful clusters based on their clinical characteristics?

**3. Association Rules**

> Which clinical conditions or characteristics frequently occur together?

---

## 🎯 Learning Objectives

Through this project, I explored:

- Distributed data processing with PySpark
- Working with data stored in HDFS
- Feature preparation for Machine Learning
- Supervised learning with Logistic Regression
- Unsupervised learning with KMeans
- Frequent pattern mining with FP-Growth
- Model evaluation and comparison
- Building a reusable ML pipeline
- Integrating a trained model into a desktop application

---

## 👨‍💻 About

**Võ Minh Huy**

Personal project focused on:

`Big Data` · `Data Mining` · `PySpark` · `Machine Learning`

---

## ⚠️ Disclaimer

This project is intended for **educational and experimental purposes**. Model predictions should not be interpreted as medical advice or used for clinical decision-making.
