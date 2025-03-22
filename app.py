from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import time

# Set Streamlit layout to wide and style the UI
st.set_page_config(layout="wide", page_title="Customer Churn Prediction")
 # Header banner

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        .prediction-result {
            font-size: 22px;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .churned {
            background-color: #ff4b4b;
            color: white;
        }
        .retained {
            background-color: #4CAF50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Information
st.sidebar.title("🌟 User Inputs")
st.sidebar.markdown("### ℹ️ About This Tool")
st.sidebar.info("This AI-powered tool predicts whether a customer will churn based on financial and demographic details. Adjust inputs and hit 'Predict'!")

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Define a mapping of user-friendly labels to actual feature names
feature_display_names = {
    "Credit Score": "CreditScore",
    "Age": "Age",
    "Tenure (Years)": "Tenure",
    "Balance Amount": "Balance",
    "Number of Products": "NumOfProducts",
    "Estimated Salary": "EstimatedSalary",
    "Geography - France": "Geography_France",
    "Geography - Germany": "Geography_Germany",
    "Geography - Spain": "Geography_Spain",
    "Gender - Female": "Gender_Female",
    "Gender - Male": "Gender_Male",
    "Has Credit Card": "HasCrCard_1",
    "No Credit Card": "HasCrCard_0",
    "Inactive Member": "IsActiveMember_0",
    "Active Member": "IsActiveMember_1",
}

# User Inputs in Sidebar
user_inputs = {}

for display_name, feature_name in feature_display_names.items():
    if feature_name in scale_vars:
        user_inputs[feature_name] = st.sidebar.number_input(
            display_name, value=600 if feature_name == "CreditScore" else 30, step=1
        )
    elif feature_name.startswith("Geography") or feature_name.startswith("Gender") or feature_name.startswith("HasCrCard") or feature_name.startswith("IsActiveMember"):
        user_inputs[feature_name] = st.sidebar.checkbox(display_name, value=True if "_1" in feature_name else False)
    else:
        user_inputs[feature_name] = st.sidebar.number_input(display_name, value=2, step=1)

# Convert inputs to DataFrame
input_data = pd.DataFrame([user_inputs])
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# Main Container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("✨ Customer Churn Prediction ✨")
st.subheader("Will the customer stay or leave? Let's find out!")

# Customer Insights Section
st.markdown("---")
st.header("📊 Customer Insights")
col1, col2 = st.columns(2)
col1.metric(label="📉 Average Credit Score", value="650", delta="↑ 5% from last year")
col2.metric(label="📈 Churn Rate", value="21.5%", delta="↓ 3% this quarter")

# Prediction Section
st.markdown("---")
st.header("🔍 Prediction")

# Ensure input_data_scaled columns match model's expected feature order
expected_feature_order = model.get_booster().feature_names
input_data_scaled = input_data_scaled[expected_feature_order]  # Reorder columns

if st.button("Predict 🚀"):
    with st.spinner("⏳ Analyzing data..."):
        time.sleep(2)  # Simulate processing time

    probabilities = model.predict_proba(input_data_scaled)[0]
    prediction = model.predict(input_data_scaled)[0]
    prediction_label = "Churned" if prediction == 1 else "Retained"

    # Styled output
    st.markdown(f"<div class='prediction-result {'churned' if prediction == 1 else 'retained'}'>"
                f"{'⚠️' if prediction == 1 else '✅'} <b>Predicted Status:</b> {prediction_label}</div>", 
                unsafe_allow_html=True)
    st.write(f"**📌 Probability of Churn:** {probabilities[1]:.2%}")
    st.write(f"**📌 Probability of Retention:** {probabilities[0]:.2%}")

st.markdown("</div>", unsafe_allow_html=True)

# FAQ Section
with st.expander("💡 What factors influence customer churn?"):
    st.write("""
    - **Credit Score**: Lower scores increase churn risk.
    - **Age & Tenure**: Younger customers with short tenures are more likely to leave.
    - **Balance & Salary**: High balance with low salary can indicate risk.
    - **Number of Products**: More products indicate customer loyalty.
    - **Activity Status**: Active members are less likely to churn.
    """)

