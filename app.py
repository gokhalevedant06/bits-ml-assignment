import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Cardiovascular Disease Classification", layout="wide")
st.title("🫀 Cardiovascular Disease Classification")

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
# SIDEBAR – MODEL SELECTION (DROPDOWN MULTISELECT)
# ============================================================
st.sidebar.header("⚙️ Controls")

model_names = list(MODEL_MAP.keys())

selected_models = st.sidebar.multiselect(
    "Select Models",
    options=model_names,
    default=model_names[0:2],
    label_visibility="visible"
)

# ---------- Custom CSS to hide red chips ----------
st.markdown(
    """
    <style>
    /* Hide selected chips inside multiselect */
    div[data-baseweb="tag"] {
        display: none !important;
    }

    /* Make dropdown look like single clean box */
    div[data-baseweb="select"] > div {
        min-height: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if not selected_models:
    st.sidebar.warning("Select at least one model.")
    st.stop()

with st.sidebar.expander("📂 Test Dataset", expanded=True):
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    use_default = st.checkbox("Use default test dataset", value=True)

    if use_default:
        st.caption("Using default test dataset. Uncheck to upload custom data.")


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    if use_default and os.path.exists(DEFAULT_TEST):
        return pd.read_csv(DEFAULT_TEST)
    if uploaded:
        return pd.read_csv(uploaded)
    return None


# ============================================================
# LOAD MODEL
# ============================================================
def load_model_and_scaler(model_name):
    model_file, needs_scaling = MODEL_MAP[model_name]

    model_path = os.path.join(MODEL_DIR, model_file)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return model, scaler, needs_scaling


# ============================================================
# TABS
# ============================================================
tab_play, tab_training, tab_readme = st.tabs(
    ["🧪 Playground", "📊 Training Insights", "📘 README"]
)


# ============================================================
# 🧪 PLAYGROUND
# ============================================================
with tab_play:

    data = load_data()

    if data is None:
        st.info("Upload or enable default dataset from sidebar.")
        st.stop()

    data = data.drop(columns=[c for c in data.columns if "Unnamed" in c], errors="ignore")

    st.info("💡 Double-click any cell to edit values in the **test dataset**.")
    edited = st.data_editor(data, use_container_width=True)

    if st.button("Run Prediction"):

        if TARGET not in edited.columns:
            st.error("Target column missing.")
            st.stop()

        X = edited.drop(columns=[TARGET])
        y = edited[TARGET]

        # ---------- Collect results ----------
        results = []

        for model_name in selected_models:

            model, scaler, needs_scaling = load_model_and_scaler(model_name)

            if model is None:
                continue

            X_input = scaler.transform(X) if (needs_scaling and scaler is not None) else X

            y_pred = model.predict(X_input)
            y_prob = model.predict_proba(X_input)[:, 1]

            results.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y, y_pred),
                "AUC": roc_auc_score(y, y_prob),
                "Precision": precision_score(y, y_pred),
                "Recall": recall_score(y, y_pred),
                "F1": f1_score(y, y_pred),
                "MCC": matthews_corrcoef(y, y_pred)
            })

        if not results:
            st.error("No models available for prediction.")
            st.stop()

        # ---------- Performance Table ----------
        st.subheader(f"📊 Model Performance Comparison : {len(selected_models)}/{len(model_names)} Models Selected")



        df_results = pd.DataFrame(results).sort_values("AUC", ascending=False)

        st.dataframe(
            df_results.style.format({
                "Accuracy": "{:.3f}",
                "AUC": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "MCC": "{:.3f}",
            }),
            use_container_width=True
        )

        # ---------- Confusion Matrices ----------
        st.subheader("Confusion Matrices")

        cols = st.columns(len(results))

        for col, res in zip(cols, results):

            model_name = res["Model"]
            model, scaler, needs_scaling = load_model_and_scaler(model_name)

            X_input = scaler.transform(X) if (needs_scaling and scaler is not None) else X
            y_pred = model.predict(X_input)

            cm = confusion_matrix(y, y_pred)

            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"]
            )

            col.write(f"**{model_name}**")
            col.dataframe(cm_df)


# ============================================================
# 📊 TRAINING INSIGHTS
# ============================================================
with tab_training:

    st.header("Insights")

    scores_path = os.path.join(MODEL_DIR, "training_scores.pkl")
    class_path = os.path.join(MODEL_DIR, "class_dist.pkl")

    if not os.path.exists(scores_path):
        st.warning("training_scores.pkl missing. Run training pipeline first.")
        st.stop()

    # ---------- Load scores ----------
    training_scores = joblib.load(scores_path)

    # ---------- Build performance table ----------
    rows = []

    for filename, scores in training_scores.items():

        # Clean model display name
        model_name = filename.replace(".pkl", "").replace("_", " ").title()

        rows.append({
            "Model": model_name,
            "Accuracy": scores.get("accuracy"),
            "AUC": scores.get("auc"),
            "Precision": scores.get("precision"),
            "Recall": scores.get("recall"),
            "F1": scores.get("f1"),
            "MCC": scores.get("mcc"),
        })

    df_scores = pd.DataFrame(rows)


    # ---------- Display ----------
    st.subheader("Model Performance Summary")
    st.dataframe(df_scores, use_container_width=True)

    st.divider()

    if os.path.exists(class_path):

        class_dist = joblib.load(class_path)   # already contains counts

        # ---- Map labels ----
        label_map = {
            0: "Healthy",
            1: "Diseased"
        }

        labels = [label_map.get(int(i), f"Class {i}") for i in class_dist.index]
        values = class_dist.values

        st.subheader("Class Distribution")

        # ---- Smaller clean figure ----
        fig, ax = plt.subplots(figsize=(4, 3))

        bars = ax.bar(labels, values)

        # ---- Add count labels on bars ----
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_xlabel("Class", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Dataset Class Distribution", fontsize=11)

        st.pyplot(fig, use_container_width=False)

    else:
        st.info("class_dist.pkl not found.")



# ============================================================
# 📘 README (FIXED VISIBILITY)
# ============================================================
with tab_readme:

    st.header("Project Documentation")

    readme_path = "README.md"

    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip():
            st.markdown(content)
        else:
            st.warning("README.md is empty.")
    else:
        st.warning("README.md file not found in project root.")
