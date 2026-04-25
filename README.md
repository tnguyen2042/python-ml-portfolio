# Python & Machine Learning Portfolio

**Languages:** Python 3.10+  
**Domain:** Healthcare analytics + benchmark datasets  
**Focus:** Applied ML for data analysts — from EDA to model deployment

---

## About

I'm a data analyst with hands-on experience in SQL and Excel, and an M.S. in Information Technology (Data Science concentration). This repository documents my progression from analyst to ML practitioner, applying concepts from graduate-level data mining coursework in Python using real datasets.

My SQL portfolio (healthcare enrollment analytics) and this Python/ML portfolio are designed as a pair — the same domain, explored at different levels of the stack.

---

## Skills Demonstrated

| Section | Folder | Key Skills |
|---|---|---|
| Exploratory Data Analysis | `01_eda/` | pandas profiling, matplotlib, seaborn, correlation analysis |
| Classification | `02_classification/` | Logistic regression, Random Forest, confusion matrix, ROC-AUC, cross-validation |
| Class Imbalance | `02_classification/` | `class_weight`, SMOTE, precision-recall tradeoffs |
| Regression | `03_regression/` | Linear, Ridge, Lasso, residual analysis, regularization |
| Visualization | `04_visualization/` | Multi-panel dashboards, storytelling, annotated charts |
| MLOps Basics | `05_mlops/` | sklearn Pipelines, joblib model saving/loading, CLI training scripts |
| Deep Learning | `06_deep_learning/` | PyTorch, feedforward neural network, MNIST digit classification |

---

## Datasets

| Dataset | Source | Used In |
|---|---|---|
| Breast Cancer Wisconsin | `sklearn.datasets.load_breast_cancer()` | EDA, Classification, Visualization, MLOps |
| Diabetes Progression | `sklearn.datasets.load_diabetes()` | Regression |
| MNIST Handwritten Digits | `torchvision.datasets.MNIST` | Deep Learning |

All datasets are loaded programmatically — no manual downloads required.

---

## How to Run

```bash
# Clone and install dependencies
git clone https://github.com/tnguyen2042/python-ml-portfolio.git
cd python-ml-portfolio
pip install -r requirements.txt

# Launch Jupyter to explore notebooks
jupyter notebook

# Run MLOps scripts directly
cd 05_mlops
python train.py
python predict.py
```

---

## Related Portfolio

[SQL Portfolio](https://github.com/tnguyen2042/sql-portfolio) — Healthcare enrollment and demographics analysis in PostgreSQL.

---

## Contact

[LinkedIn](https://linkedin.com/in/your-profile) · tnguyen.xyz01@gmail.com
