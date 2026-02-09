import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="ML Clinical Dashboard", layout="wide")
st.title("🫀 Heart Disease Prediction Dashboard")

# ============================================================
# PATHS
# ============================================================
MODEL_DIR = "models"
DEFAULT_TEST_PATH = "test.csv"
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": ("logistic_model.pkl", "scaler.pkl"),
    "Random Forest": ("rf_model.pkl", "scaler.pkl"),
    "SVM": ("svm_model.pkl", "scaler.pkl"),
}

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Controls")

use_default = st.sidebar.checkbox("Use default test dataset")

uploaded_file = None if use_default else st.sidebar.file_uploader(
    "Upload Test CSV", type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Pretrained Model",
    list(MODEL_FILES.keys())
)

# ============================================================
# LOAD MODEL
# ============================================================
def load_model_and_scaler(choice):
    model_file, scaler_file = MODEL_FILES[choice]

    model_path = os.path.join(MODEL_DIR, model_file)
    scaler_path = os.path.join(MODEL_DIR, scaler_file)

    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"Scaler not found: {scaler_path}")
        st.stop()

    return joblib.load(model_path), joblib.load(scaler_path)


model, scaler = load_model_and_scaler(model_name)

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    if use_default:
        if not os.path.exists(DEFAULT_TEST_PATH):
            st.error("Default test.csv not found.")
            st.stop()
        return pd.read_csv(DEFAULT_TEST_PATH)

    if uploaded_file:
        return pd.read_csv(uploaded_file)

    return None


data = load_data()

# ============================================================
# TOP TABS (PLAYGROUND FIRST → DEFAULT LANDING)
# ============================================================
tab_playground, tab_training, tab_readme = st.tabs(
    ["🧪 Playground", "📊 Training Insights", "📘 README"]
)

# ============================================================
# ====================== PLAYGROUND ==========================
# ============================================================
with tab_playground:

    st.header("Prediction Playground")

    if data is None:
        st.info("Upload or enable default test dataset from sidebar to begin.")
        st.stop()

    # Remove unnamed columns
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    # Editable info message
    st.success("You can edit the dataset below. Click **Run Prediction** after making changes.")

    # Editable dataframe
    edited_df = st.data_editor(
        data,
        num_rows="dynamic",
        use_container_width=True
    )

    # Run prediction
    if st.button("Run Prediction on Edited Data"):

        if TARGET_COL not in edited_df.columns:
            st.error(f"'{TARGET_COL}' column missing.")
            st.stop()

        X_test = edited_df.drop(columns=[TARGET_COL])
        y_test = edited_df[TARGET_COL]

        try:
            X_test = X_test[scaler.feature_names_in_]
        except Exception:
            st.error("Feature mismatch with training data.")
            st.stop()

        X_scaled = scaler.transform(X_test)

        # Predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # ================= DASHBOARD =================
        st.subheader("Evaluation Summary")

        left, right = st.columns([2, 1])

        with left:
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.3f}")
            c1.metric("AUC", f"{auc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c2.metric("Recall", f"{rec:.3f}")
            c3.metric("F1 Score", f"{f1:.3f}")
            c3.metric("MCC", f"{mcc:.3f}")

        with right:
            cm = confusion_matrix(y_test, y_pred)
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"]
                ),
                use_container_width=True
            )

        # ================= SAMPLE PREDICTIONS =================
        st.subheader("Sample Predictions")

        preview_df = edited_df.copy().head(10)
        preview_df["Prediction"] = y_pred[:10]

        def color_pred(val):
            return (
                "background-color: rgba(34,197,94,0.25);"
                if val == 1
                else "background-color: rgba(239,68,68,0.25);"
            )

        st.dataframe(
            preview_df.style.applymap(color_pred, subset=["Prediction"]),
            use_container_width=True
        )

# ============================================================
# =================== TRAINING INSIGHTS ======================
# ============================================================
with tab_training:

    st.header("Training Insights")

    roc_path = os.path.join(MODEL_DIR, "roc_data.pkl")
    pr_path = os.path.join(MODEL_DIR, "pr_data.pkl")
    class_path = os.path.join(MODEL_DIR, "class_dist.pkl")

    if not os.path.exists(roc_path):
        st.info("Training insight files not found. Run training save script.")
    else:
        roc_data = joblib.load(roc_path)
        pr_data = joblib.load(pr_path)
        class_dist = joblib.load(class_path)

        st.subheader("ROC Curve")
        st.line_chart(pd.DataFrame({"TPR": roc_data["tpr"]}, index=roc_data["fpr"]))

        st.subheader("Precision-Recall Curve")
        st.line_chart(pd.DataFrame({"Precision": pr_data["precision"]}, index=pr_data["recall"]))

        st.subheader("Class Distribution")
        st.bar_chart(class_dist)

# ============================================================
# ======================== README ============================
# ============================================================
with tab_readme:

    st.header("Project Documentation")

    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("README.md not found in project folder.")
