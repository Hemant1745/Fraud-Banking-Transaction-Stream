# Fraud-Banking-Transaction-Stream
Fraud Banking Transaction Stream DE mini project
# 💰 FraudSense: Bank Transaction Fraud Detection System

FraudSense is an AI-powered fraud detection system that uses a **Random Forest Classifier** to identify potentially fraudulent bank transactions.  
It integrates **Machine Learning**, **Streamlit**, and **PostgreSQL** to deliver real-time fraud predictions, analytics, and secure data management — all through an intuitive web interface.

---

## 🚀 Features

- 🔍 **Fraud Detection** using Random Forest ML model  
- ⚙️ **Real-Time Prediction** via a Streamlit web app  
- 📊 **Visual Insights** with Plotly and Matplotlib  
- 🗄️ **PostgreSQL Database** integration for secure storage  
- 🔄 **Scalable Design** for continuous model retraining and API integration  

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Machine Learning | Python, Scikit-learn (Random Forest) |
| Frontend | Streamlit |
| Database | PostgreSQL |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |

---

## 🧩 Workflow

1. **Data Collection** – Load and clean transaction dataset (CSV or DB).  
2. **Data Preprocessing** – Handle missing values, scale numeric features.  
3. **Model Training** – Train and evaluate Random Forest Classifier.  
4. **Evaluation** – Measure performance using accuracy, precision, recall, F1-score.  
5. **Integration** – Connect model with Streamlit for real-time predictions.  
6. **Visualization** – Display fraud trends and metrics using interactive charts.  
7. **Deployment** – Deploy app for real-world fraud detection use cases.  

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- PostgreSQL installed and running
- OpenWeatherMap API key *(if used for location scoring, optional)*

## Install Dependencies
pip install -r requirements.txt

## Run the Streamlit App
streamlit run app.py

| Parameter           | Description                |
| ------------------- | -------------------------- |
| Transaction Amount  | Total value of transaction |
| Account Balance     | Current account balance    |
| Transaction Hour    | Hour of transaction (0–23) |
| Customer Age        | Age of account holder      |
| Location Risk Score | Risk factor (0–10)         |

## 📈 Results

High accuracy with Random Forest classifier

Real-time prediction and visualization

Secure database storage for transactions

Easily extendable to API-based financial data

## 🧑‍💻 Contributors

Hemant Sonewale
Aadi Upadhyay 

Faculty Guide: Prof. Aditya Pai H
MIT School of Computing, Department of Computer Science & Engineering
Class: TYAIEC–2  Group ID: A6

## 🏁 Conclusion

FraudSense demonstrates the power of Machine Learning in financial security by combining intelligent classification, real-time analytics, and user-friendly visualization — enabling smarter, faster fraud detection.


