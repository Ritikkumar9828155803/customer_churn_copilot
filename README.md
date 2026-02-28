# Customer Churn Copilot

An end-to-end Machine Learning web application built with Streamlit that predicts customer churn, identifies key drivers, and suggests retention strategies.
```bash
🔗 Deployed on Streamlit Cloud
📊 Built using XGBoost and
🧠 Includes cross-validation and feature importance analysis
```
---

📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses.

This app allows users to:

-Upload a churn dataset (CSV)
-Train an XGBoost model dynamically
-Perform 5-fold cross-validation
-Predict churn probability for individual customers
-Visualize feature importance
-Estimate revenue at risk
-Generate retention strategies



---
## 📌 Features

- Upload custom churn dataset (CSV)
- Train XGBoost model dynamically
- 5-fold Cross Validation
- Churn probability prediction
- Feature importance visualization
- Revenue risk estimation
- Retention strategy suggestions

---

## 🧠 Tech Stack

-Python
-Streamlit
-XGBoost
-Scikit-learn
-Pandas
-NumPy
-Altair

---

## 📂 Project Structure
```bash
customer_churn_copilot/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Cloning Git Repository
```bash
git clone https://github.com/your-username/customer_churn_copilot.git
cd customer_churn_copilot
```

### Create Environment
```bash
python -m venv venv
```

### Activate Environment
```bash
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate  #macOS / Linux
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the App
```bash
streamlit run app.py
```

