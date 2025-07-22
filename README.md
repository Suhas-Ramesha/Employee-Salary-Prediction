# 💼 Employee Salary Prediction

👉 **[🟢 Live App (Click to Open)](https://employee-salary-prediction-gepyidmcrsbvudckwnhqk5.streamlit.app/)**

This project predicts whether an employee earns more than \$50K per year using demographic and work-related data from the UCI Adult Census dataset. The model is trained using Gradient Boosting and deployed as a user-friendly Streamlit web app.

---

## 📊 Problem Statement

Predicting employee salary brackets is a critical task for workforce planning, HR analytics, and compensation analysis. Traditional methods are manual, time-consuming, and prone to bias. This machine learning model automates the prediction process based on features like age, education, workclass, occupation, hours-per-week, and more.

---

## 🛠️ System Requirements

- Python 3.7+
- Jupyter Notebook / Google Colab
- Streamlit (for UI deployment)
- 4GB+ RAM recommended

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `joblib`
- `streamlit`

Install them via:

```bash
pip install -r requirements.txt
```

## 🧠 Model Used
The dataset is cleaned, encoded, and trained on multiple models. The best-performing model (Gradient Boosting Classifier) is selected based on accuracy and saved using Joblib.

## 🚀 Features
Predicts salary class: >50K or <=50K

Accepts manual input or CSV for batch predictions

Visual UI with real-time predictions

Model selection and evaluation included

## 📂 Project Structure
```bash
Copy code
employee-salary-prediction/
├── app.py                  # Streamlit app
├── best_model.pkl          # Trained model
├── requirements.txt        # Dependencies
├── README.md               # This file
└── data/
    └── adult.csv           # Dataset (optional)
```

## 🧪 How to Run Locally
```bash
Copy code
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📎 References
- UCI Adult Dataset

- Scikit-learn Documentation

- Streamlit Documentation

- Live App

## 📬 Contact
For questions or contributions, open an issue or reach out at: your-rsuhas319@gmail.com
