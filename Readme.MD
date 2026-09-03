# 📱 Mobile Phone Price Range Classification Using Machine Learning

## 📌 Project Overview

This project focuses on building a **multi-class machine learning classification model** to predict the price range of mobile phones based on their technical specifications.

The objective is to classify mobile phones into four price categories:

* **0 → Low Cost**
* **1 → Medium Cost**
* **2 → High Cost**
* **3 → Very High Cost**

The project follows an end-to-end machine learning workflow including **data validation, exploratory data analysis, statistical analysis, feature selection, feature engineering, model comparison, hyperparameter tuning, model evaluation, and model serialization**.

**Project ID:** PRCP-1009
**Project Type:** Supervised Machine Learning – Multi-Class Classification

---

## 🎯 Business Objective

The goal is to develop a machine learning solution that can determine the appropriate **mobile phone price category** from its hardware and feature specifications.

This type of model can help mobile manufacturers, retailers, and businesses understand how different technical specifications contribute to phone pricing and support data-driven product positioning.

---

## 🧠 Machine Learning Problem

| Item               | Description                           |
| ------------------ | ------------------------------------- |
| Problem Type       | Multi-Class Classification            |
| Target Variable    | `price_range`                         |
| Number of Classes  | 4                                     |
| Dataset Size       | 2,000 rows × 21 columns               |
| Missing Values     | None                                  |
| Duplicate Rows     | None                                  |
| Train Samples      | 1,467                                 |
| Test Samples       | 489                                   |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score |

The dataset contains 2,000 observations and 21 variables, including 20 predictor variables and the target variable `price_range`.

---

# 📊 Dataset

The dataset contains mobile phone specifications such as:

### Hardware Features

* `battery_power`
* `clock_speed`
* `int_memory`
* `mobile_wt`
* `n_cores`
* `ram`

### Camera & Display Features

* `fc` – Front Camera
* `pc` – Primary Camera
* `px_height` – Pixel Resolution Height
* `px_width` – Pixel Resolution Width
* `sc_h` – Screen Height
* `sc_w` – Screen Width

### Connectivity & Other Features

* `blue` – Bluetooth
* `dual_sim`
* `four_g`
* `three_g`
* `touch_screen`
* `wifi`
* `talk_time`
* `m_dep`

### Target

* `price_range`

The target contains four classes with an equal distribution of **500 observations per class**, making the classification problem balanced.

---

# 🔍 Exploratory Data Analysis

The project performed extensive exploratory data analysis to understand the dataset before model development.

### Data Quality

The analysis found:

* **2,000 rows**
* **21 columns**
* **0 missing values**
* **0 duplicate rows**
* 19 integer columns and 2 floating-point columns in the original dataset.

### Distribution Analysis

Skewness and kurtosis were analyzed for the numerical variables.

One important observation was:

* `fc` had a high right skewness of approximately **1.02**
* `px_height` showed moderate right skewness
* `sc_w` showed moderate right skewness
* Most remaining numerical variables were approximately symmetric.

---

# 🧹 Data Validation & Preprocessing

The project included a dedicated data validation stage.

Several discrete numerical variables were converted into categorical variables because of their limited number of unique values.

### Converted Variables

```text
blue
dual_sim
four_g
n_cores
three_g
touch_screen
wifi
price_range
```

This was done to better represent the nature of these discrete variables during analysis.

---

# 📈 Statistical Analysis

Statistical analysis was performed to investigate relationships between features and the target.

The project identified an important relationship between camera-related features:

> Front Camera (`fc`) and Primary Camera (`pc`) showed a correlation of approximately **0.6**.

This relationship was investigated using:

* Chi-Square Test
* Cramér's V
* Correlation analysis

The analysis also identified **RAM as the most influential feature** in relation to `price_range`.

---

# 🧩 Feature Selection

Two major feature-selection approaches were used.

## 1. Mutual Information

`mutual_info_classif` was used to measure the relationship between individual features and the target.

The strongest feature was:

| Feature         | Mutual Information |
| --------------- | -----------------: |
| `ram`           |       **0.853583** |
| `blue`          |           0.044889 |
| `px_width`      |           0.036194 |
| `sc_w`          |           0.033453 |
| `battery_power` |           0.030192 |
| `px_height`     |           0.029950 |
| `touch_screen`  |           0.029188 |

`ram` was substantially stronger than the other features according to the mutual-information analysis.

## 2. Random Forest Feature Importance

Random Forest was also used to investigate feature importance.

The analysis again identified **RAM as the dominant feature**, followed by variables such as:

* `battery_power`
* `px_width`
* `px_height`
* `mobile_wt`
* `int_memory`
* `pc`

This provided an additional model-based approach for understanding which phone specifications contribute most to price-range classification.

---

# 🛠️ Feature Engineering

The project included the following feature-engineering steps:

### Outlier Treatment

Outliers were investigated and the project notes that they represented less than approximately **1%** of the observations.

The **Interquartile Range (IQR)** method was used for outlier treatment.

### Feature Scaling

Numerical features were standardized using:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

Standardization was applied before models such as Logistic Regression, KNN, and SVM.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42
)
```

Result:

| Dataset  | Samples |
| -------- | ------: |
| Training |   1,467 |
| Testing  |     489 |

The split used a **75:25 train-test ratio** with `random_state=42`.

---

# 🤖 Machine Learning Models

Multiple classification algorithms were trained and compared.

### Models Evaluated

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Support Vector Classifier
5. Random Forest
6. XGBoost
7. LightGBM
8. CatBoost

This model-comparison approach helped determine which algorithm provided the strongest generalization performance on the test dataset.

---

# 📊 Model Performance

The following results were obtained from the notebook:

| Model                   | Test Accuracy | Test Precision | Test Recall |    Test F1 |
| ----------------------- | ------------: | -------------: | ----------: | ---------: |
| **Logistic Regression** |    **96.52%** |     **96.54%** |  **96.48%** | **96.50%** |
| KNN                     |        52.97% |         55.10% |      53.05% |     53.20% |
| Decision Tree           |        82.00% |         82.46% |      82.19% |     82.27% |
| SVC                     |        87.53% |         87.87% |      87.67% |     87.66% |
| Random Forest           |        88.96% |         88.95% |      88.87% |     88.87% |
| XGBoost                 |        90.59% |         90.59% |      90.55% |     90.55% |
| LightGBM                |        90.80% |         90.76% |      90.72% |     90.72% |
| CatBoost                |        93.25% |         93.25% |      93.14% |     93.15% |

Based on the notebook's model comparison, **Logistic Regression achieved the highest test F1-score among the evaluated models at approximately 96.50%**.

---

# ⚙️ Hyperparameter Tuning

Since Logistic Regression performed strongly during the initial model comparison, GridSearchCV was used to optimize its hyperparameters.

### Grid Search Configuration

```python
GridSearchCV(
    estimator=LogisticRegression(max_iter=200),
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
```

The search evaluated different combinations of:

* `C`
* `penalty`
* `solver`

### Best Parameters

```text
C = 1
penalty = l1
solver = saga
```

The best model was:

```python
LogisticRegression(
    max_iter=200,
    C=1,
    penalty='l1',
    solver='saga'
)
```

The notebook reports the tuned model's performance as:

| Metric    | Training |    Testing |
| --------- | -------: | ---------: |
| Accuracy  |   98.64% | **97.75%** |
| Precision |   98.64% | **97.78%** |
| Recall    |   98.63% | **97.73%** |
| F1-Score  |   98.63% | **97.75%** |

The tuned model therefore improved the test F1-score compared with the initial Logistic Regression model.

---

# 🏆 Final Model

### Selected Model: Logistic Regression

The final selected model is the tuned Logistic Regression classifier:

```python
LogisticRegression(
    max_iter=200,
    C=1,
    penalty='l1',
    solver='saga'
)
```

### Final Test Performance

**Accuracy: 97.75%**

**Precision: 97.78%**

**Recall: 97.73%**

**F1-Score: 97.75%**

The trained model was serialized using Python's `pickle` module and saved as:

```text
logistic_regression_model.pkl
```

This model artifact is included in the project workflow for future prediction/inference.

---

# 📋 Classification Report

The final evaluation produced strong performance across all four price categories.

| Price Class    | Precision | Recall | F1-Score |
| -------------- | --------: | -----: | -------: |
| Low Cost       |      0.98 |   0.99 |     0.99 |
| Medium Cost    |      0.98 |   0.96 |     0.97 |
| High Cost      |      0.97 |   0.97 |     0.97 |
| Very High Cost |      0.98 |   0.99 |     0.98 |

Overall accuracy was approximately **98%** on the 489 test observations.

---

# 🎯 Model Prediction Analysis

The notebook's confusion-matrix analysis showed:

* **Low Cost:** 118 of 119 samples correctly predicted
* **Medium Cost:** 110 of 115 samples correctly predicted
* **High Cost:** 125 of 128 samples correctly predicted
* **Very High Cost:** 125 of 126 samples correctly predicted

This demonstrates that the final Logistic Regression model performed consistently across all four price categories.

---

# 💡 Key Insights

### 1. RAM is the most important feature

RAM had a much stronger relationship with the target than most other variables.

Mutual Information:

```text
RAM → 0.853583
```

This makes RAM one of the strongest predictors of mobile-phone price range in this dataset.

### 2. Balanced Target

Each price category contains:

```text
500 observations
```

Therefore, the target distribution is balanced across all four classes.

### 3. Logistic Regression performed exceptionally well

The initial Logistic Regression model achieved approximately:

```text
96.52% Test Accuracy
96.50% Test F1-Score
```

After hyperparameter tuning, performance improved to:

```text
97.75% Test Accuracy
97.75% Test F1-Score
```

---

# 📊 Evaluation Techniques

The project used several techniques to evaluate and understand model performance:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* Cross-Validation
* Mutual Information
* Random Forest Feature Importance
* Statistical Tests
* Correlation Analysis
* Cramér's V

---

# 🧰 Technologies & Libraries

### Programming Language

* Python

### Data Analysis

* NumPy
* Pandas
* SciPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

### Model Persistence

* Pickle

---

# 📁 Project Structure

```text
Mobile-Phone-Price-Range-Classification-Using-Machine-Learning/
│
├── Data/
│   └── datasets_11167_15520_train.csv
│
├── Cell_Phone_Price_Prediction.ipynb
│
├── logistic_regression_model.pkl
│
├── Readme.MD
│
└── .gitignore
```

---

# 🔄 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Validation
     │
     ├── Missing Value Check
     ├── Duplicate Check
     ├── Data Type Analysis
     └── Range Validation
     │
     ▼
Exploratory Data Analysis
     │
     ├── Univariate Analysis
     ├── Distribution Analysis
     ├── Correlation Analysis
     └── Target Analysis
     │
     ▼
Statistical Analysis
     │
     ├── Chi-Square
     └── Cramér's V
     │
     ▼
Feature Selection
     │
     ├── Mutual Information
     └── Random Forest Importance
     │
     ▼
Feature Engineering
     │
     ├── Outlier Treatment
     └── Standard Scaling
     │
     ▼
Train-Test Split
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     ├── KNN
     ├── Decision Tree
     ├── SVC
     ├── Random Forest
     ├── XGBoost
     ├── LightGBM
     └── CatBoost
     │
     ▼
Model Comparison
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Final Logistic Regression Model
     │
     ▼
Model Evaluation
     │
     ▼
Model Serialization
```

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ramnallapati/Mobile-Phone-Price-Range-Classification-Using-Machine-Learning.git
```

### 2. Navigate to the project

```bash
cd Mobile-Phone-Price-Range-Classification-Using-Machine-Learning
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm catboost
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Cell_Phone_Price_Prediction.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and model-building workflow.

---

# 📌 Project Highlights

* Performed complete **EDA and data validation** on 2,000 mobile-phone records.
* Identified **RAM as the strongest predictive feature** using Mutual Information.
* Investigated feature relationships using **correlation, Chi-Square and Cramér's V**.
* Performed **IQR-based outlier treatment**.
* Applied **StandardScaler** for numerical feature normalization.
* Compared **8 machine learning classification algorithms**.
* Used **5-fold GridSearchCV** for Logistic Regression optimization.
* Improved test F1-score from approximately **96.50% to 97.75%** through hyperparameter tuning.
* Achieved approximately **97.75% test accuracy** with the final Logistic Regression model.
* Saved the trained model using **Pickle** for future inference.

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience in:

```text
Python
Data Analysis
Exploratory Data Analysis
Data Validation
Statistical Analysis
Feature Engineering
Feature Selection
Mutual Information
Correlation Analysis
Chi-Square Test
Cramér's V
Outlier Detection
Standardization
Supervised Learning
Multi-Class Classification
Model Comparison
Cross-Validation
Hyperparameter Tuning
GridSearchCV
Model Evaluation
Confusion Matrix
Model Serialization
```

---

# 👨‍💻 Author

**Rama Rao Nallapati**

Machine Learning | Data Science | AI Engineering

---

## ⭐ If you found this project useful

Feel free to ⭐ the repository and explore the notebook to understand the complete machine-learning workflow.
