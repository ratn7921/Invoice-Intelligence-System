# 🚀 Invoice Intelligence System (Fraud Detection using Machine Learning)

## 📌 Overview

The **Invoice Intelligence System** is an end-to-end Machine Learning project designed to detect potentially fraudulent or risky invoices using data-driven insights.

It integrates **SQL + Python + Scikit-learn** to build a complete pipeline from data extraction to prediction.

---

## 🎯 Features

* 🔍 Detect suspicious invoices automatically
* 📊 Feature engineering using SQL queries
* 🤖 Machine Learning model (Random Forest)
* ⚙️ Hyperparameter tuning using GridSearchCV
* 📈 Model evaluation (Accuracy, F1 Score)
* 💾 Model saving using Joblib
* 🔮 Real-time prediction capability

---

## 🏗️ Project Structure

```
Invoice-Intelligence-System/
│
├── data_preprocessing.py     # Load & preprocess data from SQLite
├── modeling_evaluation.py    # Train model + hyperparameter tuning
├── train.py                  # End-to-end training pipeline
├── models/                   # Saved ML model & scaler
├── inventory.db              # Database file
└── README.md
```

---

## 🧠 Tech Stack

* **Python**
* **SQLite**
* **Pandas & NumPy**
* **Scikit-learn**
* **Joblib**

---

## ⚙️ How It Works

1️⃣ Data is loaded from SQLite database
2️⃣ Features are engineered using SQL queries
3️⃣ Fraud labels are created based on business rules
4️⃣ Data is split into training and testing sets
5️⃣ Features are scaled using StandardScaler
6️⃣ Random Forest model is trained using GridSearchCV
7️⃣ Best model is evaluated and saved
8️⃣ Model is used for prediction

---

## 🚀 Installation & Setup

```bash
# Clone repository
git clone https://github.com/ratn7921/Invoice-Intelligence-System.git

# Navigate to project
cd Invoice-Intelligence-System

# Create virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas scikit-learn joblib
```

---

## ▶️ Run the Project

```bash
python train.py
```

---

## 📊 Sample Output

```
Random Forest Classifier Performance
----------------------------
Accuracy: 0.92

              precision    recall  f1-score
           0       0.93      0.95      0.94
           1       0.90      0.87      0.88
```

---

## 🔥 Key Highlights

✔ End-to-End ML Pipeline
✔ Real-world dataset (SQL-based)
✔ Feature Engineering + Business Logic
✔ Model Optimization using GridSearchCV
✔ Production-ready structure

---

## 💼 Use Case

* Fraud Detection in Finance
* Invoice Verification Systems
* Supply Chain Monitoring
* Risk Analysis

---

## 📈 Future Improvements

* 🌐 Web App using Flask / Streamlit
* 📊 Dashboard for visualization
* ⚡ Real-time API integration
* ☁️ Deployment on AWS / Render

---

## 👨‍💻 Author

**Ratnakar Yadav**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
