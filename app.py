import streamlit as st
import pandas as pd
import joblib

# Load model and training feature columns
model, feature_names = joblib.load("mymodel.pkl")

# Set up the page
st.set_page_config(page_title="Employee Salary Classifier", page_icon="💼", layout="centered")

st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: flex-end;
            background-color: #0e1117;
            padding: 10px 20px;
            border-bottom: 1px solid #30363d;
        }
        .navbar a {
            margin-left: 15px;
            text-decoration: none;
        }
        .navbar img {
            height: 28px;
            vertical-align: middle;
            filter: invert(100%);
        }
    </style>

    <div class="navbar">
        <a href="https://www.linkedin.com/in/parthmahadik08/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
        </a>
        <a href="https://github.com/ParthMahadik04" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
        </a>
    </div>
""", unsafe_allow_html=True)


st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Input fields (no sidebar)
age = st.slider("Age", 18, 65, 30)
education = st.selectbox("Education Level", [
    "Bachelors", "Masters", "Doctorate", "HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college", "11th", "10th"
])
occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", 
    "Armed-Forces"
])
relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"])
race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
hours_per_week = st.slider("Hours per week", 1, 80, 40)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
native_country = st.selectbox("Native Country", ["United-States", "India", "Mexico", "Philippines", "Germany", "Other"])

# Construct DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': ['Private'],  # Default for prediction
    'education': [education],
    'marital-status': ['Never-married'],  # Default for prediction
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.dataframe(input_df)

def preprocess_input(df, feature_names):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_encoded

if st.button("🔍 Predict Salary Class"):
    processed_input = preprocess_input(input_df, feature_names)
    prediction = model.predict(processed_input)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    processed_batch = preprocess_input(batch_data, feature_names)
    batch_preds = model.predict(processed_batch)

    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.dataframe(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", csv, file_name='salary_predictions.csv', mime='text/csv')
