# 📊 Telecom Customer Churn Prediction WebApp  

[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?logo=streamlit)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

---

## 📝 Project Overview  

This repository contains a **Streamlit web application** for analyzing and predicting customer churn using the popular **Telco Customer Churn dataset**.  

Customer churn—the loss of clients or subscribers—is one of the most significant challenges in the telecom industry. Retaining customers is more cost-effective than acquiring new ones, making churn prediction an essential tool for business growth.  

This project demonstrates the complete workflow of handling churn analysis, from **exploratory data analysis (EDA)** and **data wrangling** to **training machine learning models** and **predicting customer churn probabilities**.  

The app includes:  
- 📈 **EDA Dashboard**: Interactive charts to explore churn distribution and customer demographics.  
- 🔧 **Data Wrangling Tools**: Inspect missing values, data types, and descriptive statistics.  
- 🤖 **Model Training & Evaluation**: Logistic Regression, Random Forest, Gradient Boosting, and SVM with metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
- 🧮 **Prediction Interface**: Input customer details and predict churn probability.  
- 💾 **Save & Load Models**: Download trained `.pkl` files and reuse them without retraining.  

This project is designed for **business analysts, data scientists, and students** interested in understanding churn behavior and applying machine learning in real-world scenarios.  

---

## 📂 Dataset  

The app uses the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
Ensure the CSV file is available in your project directory before running.  

---

## ⚙️ Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/your-username/Telecom-Customer-Churn-Prediction-WebApp.git
cd Telecom-Customer-Churn-Prediction-WebApp
pip install -r requirements.txt
```

---

## ▶️ Running Locally  

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  

---

## 🌐 Deployment  

You can deploy this app for free using:  
- **[Streamlit Community Cloud](https://share.streamlit.io/)** (recommended)  
- [Render](https://render.com)  
- [Hugging Face Spaces](https://huggingface.co/spaces)  

---

## 📦 Requirements  

Your `requirements.txt` should include:  

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
```

---

## ✨ Features  

- Interactive **EDA** and churn distribution analysis  
- Machine learning models with performance comparison  
- **Customer churn prediction** with probability breakdown  
- Save and load **trained models** (`.pkl`)  
- Free deployment options  

---

## 📜 License  

This project is licensed under the MIT License.  
Feel free to use, modify, and distribute.  

---
