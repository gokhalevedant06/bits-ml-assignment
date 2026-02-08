import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt


# -------------------- PAGE TITLE --------------------
st.title("📊 ML Classification Evaluation App")


# -------------------- DATASET UPLOAD --------------------
st.header("a. Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())


    # -------------------- TARGET SELECTION --------------------
    target_column = st.selectbox("Select Target Column", data.columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical target if needed
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)


    # Train-test split (since only test data allowed in assignment,
    # we still split internally for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # -------------------- MODEL SELECTION --------------------
    st.header("b. Select Model")

    model_name = st.selectbox(
        "Choose a classification model",
        ["Logistic Regression", "Random Forest", "SVM"]
    )


    # Model dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    model = models[model_name]


    # -------------------- TRAIN MODEL --------------------
    if st.button("Train & Evaluate Model"):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        # -------------------- EVALUATION METRICS --------------------
        st.header("c. Evaluation Metrics")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")


        # -------------------- CONFUSION MATRIX --------------------
        st.header("d. Confusion Matrix & Classification Report")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


        # Classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))


else:
    st.info("Please upload a CSV dataset to begin.")
