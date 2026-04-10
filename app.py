# import streamlit as st
# import joblib
# import pandas as pd

# # Load artifacts
# model = joblib.load("artifacts/model.pkl")
# encoders = joblib.load("artifacts/encoder.pkl")
# columns = joblib.load("artifacts/columns.pkl")

# st.title("📊 Customer Churn Predictor")

# st.write("Enter customer details:")

# # Input fields (IMPORTANT ones only)
# tenure = st.slider("Tenure", 0, 72)
# MonthlyCharges = st.slider("Monthly Charges", 20, 120)
# TotalCharges = tenure * MonthlyCharges

# Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
# InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# # Create input dataframe
# input_dict = {
#     "tenure": tenure,
#     "MonthlyCharges": MonthlyCharges,
#     "TotalCharges": TotalCharges,
#     "Contract": Contract,
#     "InternetService": InternetService,
#     "PaymentMethod": PaymentMethod
# }

# input_df = pd.DataFrame([input_dict])

# # Add missing columns with default
# for col in columns:
#     if col not in input_df.columns:
#         input_df[col] = 0

# # Encode
# for col, le in encoders.items():
#     if col in input_df.columns:
#         try:
#             input_df[col] = le.transform(input_df[col])
#         except:
#             input_df[col] = 0

# # Reorder columns
# input_df = input_df[columns]

# if st.button("Predict"):

#     prediction = model.predict(input_df)[0]
#     prob = model.predict_proba(input_df)[0][1]

#     st.write(f"Churn Probability: {prob:.2f}")

#     if prediction == 1:
#         st.error("⚠️ Customer likely to churn")
#     else:
#         st.success("✅ Customer likely to stay")

# import streamlit as st
# import joblib
# import pandas as pd
# from datetime import datetime

# # Load artifacts
# model = joblib.load("artifacts/model.pkl")
# encoders = joblib.load("artifacts/encoder.pkl")
# columns = joblib.load("artifacts/columns.pkl")

# # Page config
# st.set_page_config(page_title="Churn Predictor", layout="wide")

# # Sidebar
# st.sidebar.title("📊 About")
# st.sidebar.info(
#     "This app predicts whether a customer is likely to churn based on key features."
# )

# # Title
# st.title("📊 Customer Churn Predictor")
# st.markdown("---")

# # Layout
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("📥 Customer Inputs")

#     tenure = st.slider("Tenure (months)", 0, 72)
#     MonthlyCharges = st.slider("Monthly Charges", 20, 120)
#     TotalCharges = tenure * MonthlyCharges

# with col2:
#     st.subheader("📡 Services")

#     Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
#     InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
#     PaymentMethod = st.selectbox(
#         "Payment Method",
#         [
#             "Electronic check",
#             "Mailed check",
#             "Bank transfer (automatic)",
#             "Credit card (automatic)"
#         ]
#     )

# # Input DataFrame
# input_dict = {
#     "tenure": tenure,
#     "MonthlyCharges": MonthlyCharges,
#     "TotalCharges": TotalCharges,
#     "Contract": Contract,
#     "InternetService": InternetService,
#     "PaymentMethod": PaymentMethod
# }

# input_df = pd.DataFrame([input_dict])

# # Encode safely
# for col, le in encoders.items():
#     if col in input_df.columns:
#         if input_df[col][0] in le.classes_:
#             input_df[col] = le.transform(input_df[col])
#         else:
#             input_df[col] = 0

# input_df = input_df[columns]

# # Prediction button
# if st.button("🔍 Predict Churn"):

#     prediction = model.predict(input_df)[0]
#     prob = model.predict_proba(input_df)[0][1]

#     st.markdown("---")
#     st.subheader("📊 Prediction Result")

#     # Metrics
#     col3, col4 = st.columns(2)

#     with col3:
#         st.metric("Churn Probability", f"{prob:.2f}")

#     with col4:
#         if prediction == 1:
#             st.error("⚠️ Likely to churn")
#         else:
#             st.success("✅ Likely to stay")

#     # Progress bar
#     st.progress(float(prob))

#     # Logging system (session-based)
#     log_entry = {
#         "Time": datetime.now().strftime("%H:%M:%S"),
#         "Tenure": tenure,
#         "MonthlyCharges": MonthlyCharges,
#         "Prediction": "Churn" if prediction == 1 else "Stay",
#         "Probability": round(prob, 2)
#     }

#     if "logs" not in st.session_state:
#         st.session_state.logs = []

#     st.session_state.logs.append(log_entry)

# # Show logs
# st.markdown("---")
# st.subheader("📜 Prediction History")

# if "logs" in st.session_state and st.session_state.logs:
#     log_df = pd.DataFrame(st.session_state.logs)
#     st.dataframe(log_df)
# else:
#     st.info("No predictions made yet.")

# # Expandable debug panel
# with st.expander("⚙️ Debug Info"):
#     st.write("Model Columns:", columns)
#     st.write("Encoded Input:", input_df)

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# ------------------ LOAD ARTIFACTS ------------------
model = joblib.load("artifacts/model.pkl")
encoders = joblib.load("artifacts/encoder.pkl")
columns = joblib.load("artifacts/columns.pkl")

# ------------------ SIDEBAR ------------------
st.sidebar.title("📊 Dashboard")
menu = st.sidebar.radio("Navigate", ["Predict", "Analytics"])

# ------------------ TITLE ------------------
st.title("🚀 Customer Churn Dashboard")

# ================== PREDICTION PAGE ==================
if menu == "Predict":

    st.subheader("📥 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72)
        MonthlyCharges = st.slider("Monthly Charges", 20, 120)
        TotalCharges = tenure * MonthlyCharges

    with col2:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # Create input dataframe
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Contract": Contract,
        "InternetService": InternetService,
        "PaymentMethod": PaymentMethod
    }

    input_df = pd.DataFrame([input_dict])

    # Encode safely
    for col, le in encoders.items():
        if col in input_df.columns:
            if input_df[col][0] in le.classes_:
                input_df[col] = le.transform(input_df[col])
            else:
                input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[columns]

    if st.button("🔍 Predict Churn"):

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Churn Probability", f"{prob:.2f}")
            st.progress(float(prob))

        with col4:
            if prediction == 1:
                st.error("⚠️ Customer likely to churn")
            else:
                st.success("✅ Customer likely to stay")

        # ------------------ EXPLAINABLE AI ------------------
        st.subheader("🧠 Why this prediction?")

        importance = pd.Series(model.coef_[0], index=columns)
        importance = importance.sort_values(key=abs, ascending=False)

        st.bar_chart(importance)

# ================== ANALYTICS PAGE ==================
elif menu == "Analytics":

    st.subheader("📈 Dataset Insights")

    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Churn distribution
    st.write("### Churn Distribution")
    churn_counts = df["Churn"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)

    # Monthly Charges vs Churn
    st.write("### Monthly Charges vs Churn")

    fig2, ax2 = plt.subplots()
    df.boxplot(column="MonthlyCharges", by="Churn", ax=ax2)
    st.pyplot(fig2)

    # Tenure Distribution
    st.write("### Tenure Distribution")

    fig3, ax3 = plt.subplots()
    df["tenure"].hist(ax=ax3)
    st.pyplot(fig3)