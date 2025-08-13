# Bayesian Classifier Analysis on Heart Disease Dataset

Overview
This project was developed as part of a school assignment in the field of Machine Learning and Data Science.  
It focuses on implementing and evaluating different Bayesian classification models on the Heart Disease dataset, including:

- **APrioriClassifier** – predicts based on prior probability.
- **ML2DClassifier** – Maximum Likelihood estimation using conditional probabilities from one attribute.
- **MAP2DClassifier** – Maximum A Posteriori estimation combining priors and conditional probabilities.
- **MLNaiveBayesClassifier** – Naive Bayes with maximum likelihood estimation.
- **MAPNaiveBayesClassifier** – Naive Bayes with MAP estimation.
- **ReducedMLNaiveBayesClassifier** – Naive Bayes with feature selection based on Chi-square independence tests.
- **ReducedMAPNaiveBayesClassifier** – MAP Naive Bayes with feature selection.

The project also includes **data discretization**, **mutual information analysis**, and **graph-based feature dependency visualization**.

---

Dataset
Data files:
- `data/heart.csv` – full dataset
- `data/train.csv` – training subset
- `data/test.csv` – test subset

---

Features Implemented
- **Data preprocessing & discretization**
- **Prior probability estimation** with 95% confidence intervals
- **Conditional probability computation** in both directions (P(attr | target) and P(target | attr))
- **Model performance evaluation** using precision and recall
- **Feature independence testing** with Chi-square
- **Naive Bayes model visualization** using Graphviz
- **Mutual information & conditional mutual information matrices**
- **Graph algorithms (Kruskal)** for dependency structure learning
- **Comparison plots** of multiple classifiers in precision-recall space

---

Example Results
Precision vs Recall plots are generated to visually compare classifiers:  
- Ideal models appear in the **top-right corner** (high precision & recall)
- Trade-offs between recall and precision are discussed

---

Technologies Used
- **Python** (pandas, NumPy, Matplotlib, seaborn)
- **scikit-learn** (RandomForestClassifier, GaussianNB, train/test split)
- **SciPy** (Chi-square tests, statistical functions)
- **pydot** & **Graphviz** for model visualization

---
