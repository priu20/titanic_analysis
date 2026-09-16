# Titanic Survival Analysis 🚢

A data science project predicting passenger survival on the Titanic using exploratory data analysis, feature engineering, and machine learning.

## 📌 Overview

This project explores the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) to understand what factors influenced passenger survival, and builds a Random Forest classifier to predict survival outcomes.

## 📊 Dataset

- **Source:** Kaggle Titanic: Machine Learning from Disaster
- **Size:** 891 passengers (training set)
- **Target variable:** `Survived` (0 = Died, 1 = Survived)
- **Features:** Passenger class, sex, age, fare, family relationships, embarkation port, and more

## 🔍 Approach

1. **Exploratory Data Analysis (EDA)** — visualized survival patterns across sex, passenger class, age, and family size using Seaborn/Matplotlib
2. **Data Cleaning** — handled missing values in `Age` (imputed by Pclass + Sex median), `Embarked` (mode), and dropped/encoded `Cabin`
3. **Feature Engineering**
   - `Title` — extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
   - `FamilySize` — combined siblings/spouses + parents/children + self
   - `IsAlone` — flag for solo travelers
4. **Encoding** — one-hot encoded categorical variables (`Sex`, `Embarked`, `Title`)
5. **Modeling** — trained a Random Forest Classifier, then tuned hyperparameters with `GridSearchCV`
6. **Evaluation** — accuracy, precision/recall, confusion matrix, and 5-fold cross-validation

## 📈 Key Findings

- **Sex** was the single strongest predictor — women survived at ~74% vs ~19% for men
- **Passenger class** showed a clear gradient: 1st class (63%) → 2nd (47%) → 3rd (24%)
- **Title** (extracted from names) captured survival signal beyond raw Sex/Age — e.g. "Master" (young boys) survived far more than "Mr"
- **Family size** had a non-linear effect — small families (2-4 people) survived more than solo travelers or very large families
- Passengers who embarked at **Cherbourg (C)** had notably higher survival, largely as a proxy for wealth/class

## 🤖 Model Performance

| Model | Validation Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| Random Forest (baseline) | 83.8% | — |
| Random Forest (tuned) | 81.6% | 83.3% |
| Logistic Regression (tuned) | 83.2% | 81.2% |

**Top predictive features:** Sex, Title (Mr/Miss/Mrs), Fare, Passenger Class

## 🛠️ Tech Stack

- Python (pandas, numpy)
- scikit-learn (RandomForestClassifier, GridSearchCV, LogisticRegression)
- Matplotlib & Seaborn (visualization)
- Jupyter/Google Colab

## 📁 Repository Structure

```
titanic-survival-analysis/
├── titanic_survival_analysis.ipynb   # Main notebook (EDA, feature engineering, modeling)
├── train.csv                         # Training dataset
├── README.md                         # Project overview (this file)
└── requirements.txt                  # Python dependencies
```

## 🚀 How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/YOUR_USERNAME/titanic-survival-analysis.git
   cd titanic-survival-analysis
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Open `titanic_survival_analysis.ipynb` in Jupyter Notebook or upload it to Google Colab

## 📝 Future Improvements

- Try gradient boosting models (XGBoost, LightGBM) for potentially higher accuracy
- Build an interactive Power BI dashboard for exploratory storytelling
- Deploy the trained model as a simple web app for live predictions

## 📄 License

This project is open source and available under the MIT License.

---

*Dataset source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)*

