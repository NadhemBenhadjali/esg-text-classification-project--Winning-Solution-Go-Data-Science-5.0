# ESG Text Classification Project

A complete **GitHub-ready solution for ESG multi-label text classification**, including exploratory analysis, classical NLP baselines, transformer models, and a deployable demo application.

This repository demonstrates an **end-to-end machine learning workflow**, from data exploration to model deployment.

---

# Overview

This project packages the full workflow for solving a **multi-label ESG text classification problem**.

Each text document may contain one or more of the following labels:

- **E — Environmental**
- **S — Social**
- **G — Governance**
- **Non-ESG**

The repository includes:

- 📊 **EDA Notebook** with class distribution, label co-occurrence, outlier detection, and word clouds  
- 📚 **Classical NLP Baseline** using TF-IDF + Logistic Regression  
- 🤖 **Transformer Baseline** using DeBERTa for multi-label classification  
- 🖥 **Enterprise Demo App** built with Gradio  
- 📈 **EDA Assets** including plots, summary CSVs, and a PDF report

---

# Project Structure

```

text-esg-text-classification-project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── notebooks/
│   ├── ESG_EDA_source.ipynb
│   └── ESG_EDA_executed.ipynb
│
├── reports/
│   └── EDA_report.pdf
│
├── assets/
│   ├── eda/
│   │   ├── label_distribution.csv
│   │   ├── label_cooccurrence.csv
│   │   └── outlier_summary_text_features.csv
│   │
│   └── plots/
│       ├── label_frequency.png
│       ├── label_cardinality.png
│       ├── label_cooccurrence_heatmap.png
│       ├── boxplot_*.png
│       └── wordcloud_*.png
│
├── src/
│   ├── app/
│   │   └── demo_app.py
│   │
│   ├── baselines/
│   │   └── classical_tfidf_logreg.py
│   │
│   └── models/
│       └── transformer_multilabel.py

```

---

# Main Components

## 1. Exploratory Data Analysis (EDA)

The EDA highlights several key modeling challenges:

### Class Imbalance
- The **Environmental (E)** label is significantly rarer than the others.

### Noise
- The **Non-ESG** class contains many ESG-like business terms.

### Outliers
Some texts are:

- Extremely long
- Boilerplate-heavy
- URL-heavy

Outputs include:

- Label frequency plots
- Co-occurrence heatmaps
- Text length distributions
- Word clouds

Notebook:

```

notebooks/ESG_EDA_executed.ipynb

```

Report:

```

reports/EDA_report.pdf

```

---

# 2. Classical NLP Baseline

File:

```

src/baselines/classical_tfidf_logreg.py

```

Features:

- Word + Character **TF-IDF**
- **One Logistic Regression model per label**
- **Stratified K-Fold cross-validation**
- Label-combination bucketing
- **Fold-averaged predictions**

This baseline provides a **fast and interpretable benchmark**.

---

# 3. Transformer Baseline

File:

```

src/models/transformer_multilabel.py

```

Features:

- **DeBERTa-based encoder**
- Multi-label classification head
- **Sigmoid outputs**
- **BCEWithLogitsLoss**
- Class imbalance handled using **pos_weight**
- Stratified K-Fold cross-validation
- Fold-averaged predictions for final submission

This model significantly improves performance compared to classical approaches.

---

# 4. Demo Application

File:

```

src/app/demo_app.py

````

A **Gradio-based enterprise demo** that simulates a real ESG workflow:

Example workflow:

1. Text ingestion
2. ESG classification
3. Ticket triage
4. Routing and ownership
5. Urgency / SLA logic
6. Batch queue building
7. Reviewer feedback loop
8. Weekly executive reporting

This demonstrates how the model can be integrated into **real enterprise processes**.

---

# Installation

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

---

# Running the Project

## Run the Classical Baseline

```bash
python src/baselines/classical_tfidf_logreg.py
```

---

## Run the Transformer Model

```bash
python src/models/transformer_multilabel.py
```

---

## Launch the Demo App

```bash
python src/app/demo_app.py
```

---

## Open the EDA Notebook

```bash
jupyter notebook notebooks/ESG_EDA_executed.ipynb
```

---

# Notes

* Some scripts may still contain **path constants from earlier runs**.
  Update paths if necessary for your local environment.

* **Raw competition data is intentionally excluded** from the repository.

* The **executed notebook is included** for quick viewing.

---

# Suggested Next Improvements

Potential upgrades:

* Per-label threshold tuning
* Model ensembling

  * ModernBERT
  * BERT
  * DeBERTa-large
* Experiment tracking (MLflow / Weights & Biases)
* Training configuration files
* Shared configuration module
* Dataset versioning
* CI/CD for model evaluation

---

# Tech Stack

* Python
* Scikit-learn
* PyTorch
* HuggingFace Transformers
* Gradio
* Pandas
* Matplotlib / Seaborn

---

# Project Goal

This repository demonstrates a **production-ready NLP pipeline** that bridges:

* **Data Science**
* **Machine Learning Engineering**
* **Model Deployment**

---

# License

MIT License

---

# Author

Winning Solution — Go Data Science 5.0

```
which makes the repo **much more impressive to recruiters**.
```
