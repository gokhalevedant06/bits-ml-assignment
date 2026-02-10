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
# CONFIG
# ============================================================
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
st.title("🫀 Heart Disease Prediction Dashboard")

MODEL_DIR = "models"
DEFAULT_TEST = "test.csv"
TARGET = "target"

MODEL_MAP = {
    "Logistic Regression": ("logistic_model.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "KNN": ("knn.pkl", True),
    "Naive Bayes": ("naive_bayes.pkl", True),
    "Random Forest": ("random_forest.pkl", False),
    "XGBoost": ("xgboost.pkl", True),
}

# ============================================================
# SIDEBAR
# ============================================================
use_default = st.sidebar.checkbox("Use default test dataset")
uploaded = None if use_default else st.sidebar.file_uploader("Upload CSV", type=["csv"])
model_name = st.sidebar.selectbox("Select Model", list(MODEL_MAP.keys()))

# ============================================================
# LOAD MODEL
# ============================================================
model_file, needs_scaling = MODEL_MAP[model_name]

model_path = os.path.join(MODEL_DIR, model_file)
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model or scaler missing inside models/ folder.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    if use_default and os.path.exists(DEFAULT_TEST):
        return pd.read_csv(DEFAULT_TEST)
    if uploaded:
        return pd.read_csv(uploaded)
    return None

data = load_data()

# ============================================================
# TABS (PLAYGROUND FIRST → DEFAULT)
# ============================================================
tab_play, tab_training, tab_readme = st.tabs(
    ["🧪 Playground", "📊 Training Insights", "📘 README"]
)

# ============================================================
# 🧪 PLAYGROUND
# ============================================================
with tab_play:

    st.header("Prediction Playground")

    if data is None:
        st.info("Upload or enable default dataset from sidebar.")
        st.stop()

    data = data.drop(columns=[c for c in data.columns if "Unnamed" in c], errors="ignore")

    st.success("Dataset is editable. Modify values and click **Run Prediction**.")

    edited = st.data_editor(data, use_container_width=True)

    if st.button("Run Prediction"):

        if TARGET not in edited.columns:
            st.error("Target column missing.")
            st.stop()

        X = edited.drop(columns=[TARGET])
        y = edited[TARGET]

        if needs_scaling:
            X = scaler.transform(X)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        left, right = st.columns([2, 1])

        with left:
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.3f}")
            c1.metric("AUC", f"{auc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c2.metric("Recall", f"{rec:.3f}")
            c3.metric("F1", f"{f1:.3f}")
            c3.metric("MCC", f"{mcc:.3f}")

        with right:
            cm = confusion_matrix(y, y_pred)
            st.dataframe(pd.DataFrame(cm))

# ============================================================
# 📊 TRAINING INSIGHTS (ROBUST AUTO-INIT)
# ============================================================
with tab_training:

    st.header("Training Insights")

    summary_path = os.path.join(MODEL_DIR, "training_summary.pkl")
    scores_path = os.path.join(MODEL_DIR, "training_scores.pkl")

    # ---------- Ensure training_scores exists ----------
    if not os.path.exists(scores_path):
        st.error("training_scores.pkl missing in models/.")
        st.info("Run training pipeline once to generate evaluation metrics.")
        st.stop()

    training_scores = joblib.load(scores_path)

    # ---------- Auto-generate training summary ----------
    if not os.path.exists(summary_path):

        st.warning("training_summary.pkl missing → generating automatically...")

        summary = {}

        for name, (file, _) in MODEL_MAP.items():

            model_file_path = os.path.join(MODEL_DIR, file)

            if not os.path.exists(model_file_path):
                continue

            model_obj = joblib.load(model_file_path)

            params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}

            convergence = "Trained successfully"
            if "max_iter" in params:
                convergence = f"Max iterations = {params['max_iter']}"

            if hasattr(model_obj, "n_iter_"):
                try:
                    convergence = f"Converged in {model_obj.n_iter_} iterations"
                except Exception:
                    pass

            scores = training_scores.get(file, {})

            summary[name] = {
                "hyperparameters": params,
                "convergence": convergence,
                "scores": scores,
            }

        joblib.dump(summary, summary_path)
        st.success("training_summary.pkl generated successfully.")

    # ---------- Load summary ----------
    summary = joblib.load(summary_path)

    if not summary:
        st.error("Training summary is empty.")
        st.stop()

    # ---------- Display table ----------
    rows = []
    for model_name, info in summary.items():
        scores = info.get("scores", {})
        rows.append({
            "Model": model_name,
            "Accuracy": scores.get("accuracy"),
            "AUC": scores.get("auc"),
            "Precision": scores.get("precision"),
            "Recall": scores.get("recall"),
            "F1": scores.get("f1"),
            "Convergence": info.get("convergence"),
        })

    st.subheader("Model Performance Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # ---------- ROC / PR / Class Distribution ----------
    roc_path = os.path.join(MODEL_DIR, "roc_data.pkl")
    pr_path = os.path.join(MODEL_DIR, "pr_data.pkl")
    class_path = os.path.join(MODEL_DIR, "class_dist.pkl")

    if os.path.exists(roc_path) and os.path.exists(pr_path):
        roc_data = joblib.load(roc_path)
        pr_data = joblib.load(pr_path)

        st.subheader("ROC Curve")
        st.line_chart(pd.DataFrame({"TPR": roc_data["tpr"]}, index=roc_data["fpr"]))

        st.subheader("Precision-Recall Curve")
        st.line_chart(pd.DataFrame({"Precision": pr_data["precision"]}, index=pr_data["recall"]))
    else:
        st.warning("ROC / PR data not found.")

    if os.path.exists(class_path):
        class_dist = joblib.load(class_path)
        st.subheader("Class Distribution")
        st.bar_chart(class_dist)

# ============================================================
# 📘 README
# ============================================================
with tab_readme:

    if os.path.exists("README.md"):
        st.markdown(open("README.md", encoding="utf-8").read())
    else:
        st.info("README.md not found.")
