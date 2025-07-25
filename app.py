import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (must match training features)
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Others"
])
fnlwgt = st.sidebar.number_input("fnlwgt (final weight)", min_value=10000, max_value=1000000, value=100000)
educational_num = st.sidebar.slider("Educational-num (years)", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Others"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
    "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
    "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
    "Peru", "Hong", "Holand-Netherlands"
])

# Label encoding mappings (must match those used in training)
workclass_map = {'Private': 4, 'Self-emp-not-inc': 5, 'Self-emp-inc': 2, 'Federal-gov': 0, 'Local-gov': 3, 'State-gov': 6, 'Others': 1}
marital_status_map = {'Married-civ-spouse': 2, 'Divorced': 0, 'Never-married': 3, 'Separated': 4, 'Widowed': 5, 'Married-spouse-absent': 1, 'Married-AF-spouse': 6}
occupation_map = {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 7, 'Sales': 10, 'Exec-managerial': 3, 'Prof-specialty': 8, 'Handlers-cleaners': 4,
                  'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 5, 'Transport-moving': 13, 'Priv-house-serv': 9, 'Protective-serv': 11, 'Armed-Forces': 1, 'Others': 14}
relationship_map = {'Wife': 5, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 0}
race_map = {'White': 4, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3, 'Black': 2}
gender_map = {'Male': 1, 'Female': 0}
native_country_map = {name: i for i, name in enumerate([
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
    "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
    "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
    "Peru", "Hong", "Holand-Netherlands"
])}

# Build input DataFrame
input_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass_map[workclass],
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status_map[marital_status],
    'occupation': occupation_map[occupation],
    'relationship': relationship_map[relationship],
    'race': race_map[race],
    'gender': gender_map[gender],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country_map[native_country]
}])

st.write("### 🔎 Input Data (encoded)")
st.write(input_df)

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction (user must upload a CSV with the same columns as above, encoded)
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction (encoded columns)", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
