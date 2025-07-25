import streamlit as st
import pandas as pd
import joblib

# Add custom CSS for dark gradient background and improved readability
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
    }
    .stSidebar {
        background: rgba(30, 30, 40, 0.95) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.18) !important;
        color: #fff !important;
    }
    .st-bb, .st-c0, .st-c1, .st-c2, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9 {
        background: rgba(40, 40, 60, 0.92) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.18) !important;
        color: #fff !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #434343 0%, #262626 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5em 2em !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #262626 0%, #434343 100%) !important;
        color: #fff !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        color: #fff !important;
    }
    .stMarkdown, .stDataFrame, .stTable {
        background: rgba(40, 40, 60, 0.92) !important;
        border-radius: 12px !important;
        padding: 1em !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.16) !important;
        color: #fff !important;
    }
    .css-1v0mbdj, .css-1d391kg, .css-1cpxqw2, .css-1offfwp, .css-1r6slb0, .css-1kyxreq, .css-1dp5vir, .css-1v4eu6x {
        color: #fff !important;
    }
    label, .stSlider, .stSelectbox, .stNumberInput, .stFileUploader, .stTextInput {
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (minimal)
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Others"
])
educational_num = st.sidebar.slider("Educational-num (years)", 5, 16, 10)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Others"
])
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
occupation_map = {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 7, 'Sales': 10, 'Exec-managerial': 3, 'Prof-specialty': 8, 'Handlers-cleaners': 4,
                  'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 5, 'Transport-moving': 13, 'Priv-house-serv': 9, 'Protective-serv': 11, 'Armed-Forces': 1, 'Others': 14}
native_country_map = {name: i for i, name in enumerate([
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
    "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
    "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
    "Peru", "Hong", "Holand-Netherlands"
])}

# Default values for other features
fnlwgt = 100000
marital_status = 2  # e.g., 'Married-civ-spouse'
relationship = 3    # e.g., 'Not-in-family'
race = 4            # e.g., 'White'
gender = 1          # e.g., 'Male'
capital_gain = 0
capital_loss = 0

# Build input DataFrame in the order expected by the model
input_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass_map[workclass],
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation_map[occupation],
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country_map[native_country]
}])

st.write("### 🔎 Input Data (encoded, used for prediction)")
st.write(input_df)
st.write(f"**Years of Experience (not used in prediction):** {experience}")

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
