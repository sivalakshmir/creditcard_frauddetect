import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from main import load_data, train_model

st.title("Credit Card Fraud Detection App")

# Load data and train model
X, Y = load_data()
model, accuracy = train_model(X, Y)
st.success(f"Model trained with accuracy: {accuracy:.2f}")

# 1. Class Distribution
st.subheader("Class Distribution")
class_counts = Y.value_counts()
fig, ax = plt.subplots()
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
ax.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
ax.set_ylabel("Number of Transactions")
st.pyplot(fig)

# 2. Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
corr = X.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 3. Amount vs Class Boxplot
st.subheader("Transaction Amounts by Class")
df = pd.read_csv("creditcard.csv")
fig, ax = plt.subplots()
sns.boxplot(x='Class', y='Amount', data=df, ax=ax)
ax.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
st.pyplot(fig)

# 4. Confusion Matrix
st.subheader("Confusion Matrix")
Y_pred = model.predict(X)
cm = confusion_matrix(Y, Y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 5. ROC Curve and AUC
st.subheader("ROC Curve")
y_score = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(Y, y_score)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend()
st.pyplot(fig)

# User input for prediction
st.markdown("---")
st.header("Predict Fraud for a New Transaction")

features = X.columns.tolist()
input_data = []

with st.form("fraud_prediction_form"):
    for feature in features:
        value = st.number_input(f"{feature}", value=0.0, step=0.1)
        input_data.append(value)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "Fraudulent" if prediction == 1 else "Legitimate"
    st.subheader(f"Prediction: **{result}**")
