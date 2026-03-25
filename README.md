# 🏦 Credit Risk Classification System

An end-to-end Machine Learning solution designed to predict the probability of loan default. This project translates raw financial, demographic, and credit bureau data into actionable risk scores using a robust MLOps pipeline.

## 🚀 Project Overview
The primary goal of this system is to assist financial institutions in making data-driven lending decisions. By analyzing historical data from customers, loans, and credit bureaus, the model identifies patterns that correlate with high-risk behavior.

---

## 🛠️ Algorithms & Techniques Used

### 1. Model Selection: Logistic Regression
Chosen for its high interpretability in the financial sector. 
* **Solver:** `SAGA` — selected to handle large datasets and L1/L2 regularization efficiently.
* **Probability Output:** The model doesn't just classify (Default vs. Non-Default); it provides a **Probability Score**, allowing for custom "Risk Thresholds" based on the bank's risk appetite.

### 2. Advanced Feature Engineering
* **Multi-Source Integration:** Merged three distinct datasets (Customer, Loan, and Bureau) to create a holistic view of the applicant.
* **Derived Metrics:** Engineered key financial ratios including:
    * `Loan-to-Income Ratio`: Measuring debt burden relative to earning capacity.
    * `Credit Utilization`: Tracking how much available credit a user actually consumes.
    * `Delinquency Ratio`: Quantifying past payment behavior reliability.

### 3. Data Preprocessing & Scaling
* **One-Hot Encoding:** Converted categorical variables (Employment Status, Loan Purpose, Residence Type) into numerical formats.
* **Min-Max Scaling:** Applied to numerical features to ensure that variables with larger magnitudes (like Income) do not disproportionately bias the model weights.

### 4. Handling Class Imbalance
* **Stratified Splitting:** Implemented to ensure the ratio of "Default" to "Non-Default" cases remains consistent across training and testing sets, preventing model bias toward the majority class.

---

## 📊 Model Performance
The model was evaluated on a hold-out test set (20% of the total data). Below are the results for identifying "High Risk" applicants:

| Metric | Value |
|:---|:---|
| **Accuracy** | 88.45% |
| **Precision (Default)** | 84.12% |
| **Recall (Default)** | 79.80% |
| **F1-Score** | 81.90% |



---

## 📉 Feature Importance Analysis

To evaluate the predictive power of the model, we analyzed the standardized coefficients. This visualization shows which factors most heavily influence a "Default" prediction.

![Top 10 Feature Coefficients](artifacts/model_performances.png)  

### **Key Insights:**
* **Bureau Data is King:** The strongest predictors are `delinquency_ratio` and `avg_dpd_per_delinquency`. Higher values here correlate significantly with higher risk.
* **Debt Burden:** `loan_to_income` and `credit_utilization_ratio` exhibit positive coefficients, proving that high debt relative to income is a primary risk indicator.
* **Protective Factors:** Features like `income` and `number_of_open_accounts` have negative coefficients, suggesting they serve as indicators of financial stability that lower the risk profile.

---

## 💻 Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
* **Visualization:** Matplotlib, Seaborn
* **Frontend:** Streamlit (Interactive Web Application)

---

## 📂 Project Structure
```text
├── artifacts/               # Saved model and scaler (.joblib)
├── dataset/                 # Raw data (Customers, Loans, Bureau)
├── Credit_Risk_Model.ipynb  # EDA & Model Training
├── prediction_helper.py     # Backend processing logic
└── main.py                  # Streamlit application entry point




