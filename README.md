# 🚀 Customer Churn Copilot

An end-to-end ML-powered Streamlit application that predicts customer churn, explains key drivers, and generates retention strategies.

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

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy

---

## 📂 Project Structure

customer_churn_project/
│
├── app.py
├── churn_data.csv
├── requirements.txt
├── README.md
├── .gitignore
└── models/

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd customer_churn_project
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
