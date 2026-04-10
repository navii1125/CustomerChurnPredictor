import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from preprocess import preprocess_data

print("STARTING TRAINING...")

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Data loaded successfully")

# ✅ Select ONLY features used in app
selected_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "Churn"
]

df = df[selected_features]

# Preprocess
df, encoders = preprocess_data(df)
print("Data preprocessed")

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Model trained")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("ROC-AUC:", roc)

# Save artifacts
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(encoders, "artifacts/encoder.pkl")
joblib.dump(X.columns.tolist(), "artifacts/columns.pkl")

print("Artifacts saved successfully")