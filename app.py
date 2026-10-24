"""
app.py
------
AI-Powered Cyber Threat Detection and Security Analytics System
Streamlit Dashboard — entry point.

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import (
    load_csv,
    detect_label_column,
    clean_dataframe,
    prepare_features,
    get_dataset_summary,
    validate_prediction_input,
    LABEL_BENIGN,
    LABEL_ATTACK,
)
from ml_pipeline import CyberThreatModel, SUPPORTED_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber Threat Detection | AI Security Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — professional cybersecurity theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
}
[data-testid="stSidebar"] .stRadio label {
    color: #c9d1d9 !important;
    font-size: 0.95rem;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
    color: #c9d1d9 !important;
}
/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #8b949e !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important; font-weight: 700;
}
/* ── Cards / info boxes ── */
.info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    color: #c9d1d9;
}
.warning-card {
    background: #1c1209;
    border: 1px solid #9e6a03;
    border-left: 4px solid #e3b341;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    color: #e3b341;
}
.attack-badge {
    background: #3d0000;
    border: 1px solid #f85149;
    border-radius: 6px;
    color: #f85149;
    display: inline-block;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.85rem;
}
.benign-badge {
    background: #003d1a;
    border: 1px solid #3fb950;
    border-radius: 6px;
    color: #3fb950;
    display: inline-block;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.85rem;
}
/* ── Section headings ── */
h2, h3 { color: #58a6ff; }
/* ── Divider ── */
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "X": None,
        "y": None,
        "label_col": None,
        "clean_report": {},
        "model": None,
        "metrics": {},
        "feature_cols": [],
        "dataset_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CyberGuard AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "📊  Dataset Explorer",
            "🤖  Model Training",
            "🔍  Threat Detection",
            "ℹ️  Model Info",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.78rem;color:#6e7681;'>"
        "⚠️ Educational use only.<br>"
        "Not a production security tool.<br><br>"
        "Mugdha Rahate — Cybersecurity Internship Project"
        "</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")


def _load_sample_dataset():
    """Load the bundled synthetic sample dataset into session state."""
    try:
        df = load_csv(SAMPLE_DATA_PATH)
        _store_dataset(df, "sample_data.csv (synthetic demo)")
        st.success("✅ Synthetic sample dataset loaded for demonstration.")
        st.info(
            "⚠️ This is **synthetic data** created for demonstration purposes only. "
            "It does not represent real network traffic and results cannot be used "
            "to evaluate real-world detection performance."
        )
    except Exception as exc:
        st.error(f"Could not load sample dataset: {exc}")


def _store_dataset(df: pd.DataFrame, name: str):
    """Validate, clean, and store a dataset in session state."""
    label_col = detect_label_column(df)
    if label_col is None:
        st.error(
            "❌ No label column found. The dataset must contain a column named "
            "'Label', 'Class', or 'Target' with BENIGN / attack values."
        )
        return

    df_clean, report = clean_dataframe(df)
    X, y = prepare_features(df_clean, label_col)

    if X.shape[1] == 0:
        st.error("❌ No numeric feature columns found after preprocessing.")
        return

    st.session_state["df_raw"] = df
    st.session_state["df_clean"] = df_clean
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["label_col"] = label_col
    st.session_state["clean_report"] = report
    st.session_state["dataset_name"] = name
    # Reset model when new data is loaded
    st.session_state["model"] = None
    st.session_state["metrics"] = {}
    st.session_state["feature_cols"] = []


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown(
        "<h1 style='color:#58a6ff;margin-bottom:4px;'>🛡️ AI-Powered Cyber Threat Detection</h1>"
        "<h3 style='color:#8b949e;font-weight:400;margin-top:0;'>Security Analytics System</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
### About This System

This educational project demonstrates how **Machine Learning** can be applied to
**network traffic analysis** to detect potential cyber threats.

The system takes network flow features (as CSV) and classifies each flow as
**BENIGN** or **ATTACK** using a trained Random Forest (or alternative) classifier.

---
#### 🔑 Key Capabilities

| Capability | Description |
|---|---|
| 📂 Dataset Explorer | Upload and inspect any CIC-IDS2017-format CSV |
| 🧠 Model Training | Train a classifier and view real evaluation metrics |
| 🔍 Threat Detection | Upload flows for batch prediction and download results |
| 📈 Visualisations | Confusion matrix, feature importances, class distribution |
| 💾 Model Persistence | Save and reload trained models |

---
#### 📚 Recommended Dataset

This system is designed for the **CIC-IDS2017** dataset from the Canadian
Institute for Cybersecurity.

**To obtain the data:**
1. Visit [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Register and download one of the processed CSV files (e.g. Monday-WorkingHours.pcap_ISCX.csv)
3. Upload it in the **Dataset Explorer** tab

A small **synthetic sample** dataset is bundled for immediate demonstration.
""")

    with col2:
        st.markdown("""
<div class="warning-card">
<strong>⚠️ Educational Disclaimer</strong><br><br>
This is an <strong>educational internship project</strong>.<br><br>
• Does <strong>not</strong> perform live network monitoring<br>
• Does <strong>not</strong> scan or access any devices<br>
• Does <strong>not</strong> take offensive security actions<br>
• Synthetic-data results do <strong>not</strong> reflect real-world detection rates<br><br>
Use responsibly and only on data you are authorised to analyse.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="info-card">
<strong>🚀 Quick Start</strong><br><br>
1. Go to <em>Dataset Explorer</em><br>
2. Upload a CSV or load the sample data<br>
3. Go to <em>Model Training</em> and click Train<br>
4. Use <em>Threat Detection</em> to classify new flows
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='color:#6e7681;font-size:0.85rem;'>"
        "Built with Python · scikit-learn · Streamlit · pandas · Matplotlib · Seaborn"
        "</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  Dataset Explorer":
    st.markdown("<h2>📊 Dataset Explorer</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # Upload / sample buttons
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a network traffic CSV (CIC-IDS2017 format or compatible)",
            type=["csv"],
            help="Must include a 'Label' column with BENIGN / attack values.",
        )
        if uploaded is not None:
            with st.spinner("Loading and cleaning dataset…"):
                try:
                    df_raw = load_csv(uploaded)
                    _store_dataset(df_raw, uploaded.name)
                    if st.session_state["df_clean"] is not None:
                        st.success(f"✅ Dataset '{uploaded.name}' loaded successfully.")
                except Exception as exc:
                    st.error(f"Failed to load CSV: {exc}")

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📦 Load Sample Dataset", use_container_width=True):
            _load_sample_dataset()

    # ── Display dataset info ──────────────────────────────────────────────────
    df_clean = st.session_state["df_clean"]
    if df_clean is None:
        st.info("👆 Upload a CSV file or load the sample dataset to get started.")
        st.stop()

    df_raw = st.session_state["df_raw"]
    y = st.session_state["y"]
    X = st.session_state["X"]
    report = st.session_state["clean_report"]
    label_col = st.session_state["label_col"]

    st.markdown(
        f"**Dataset:** `{st.session_state['dataset_name']}`  •  "
        f"**Label column:** `{label_col}`"
    )

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 Total Rows", f"{len(df_clean):,}")
    c2.metric("🔢 Features", f"{X.shape[1]:,}")
    c3.metric("🗑️ Duplicates Removed", f"{report.get('duplicates_removed', 0):,}")
    c4.metric("🔧 Missing Imputed", f"{report.get('missing_values_imputed', 0):,}")
    c5.metric("♾️ Inf Values Fixed", f"{report.get('inf_values_replaced', 0):,}")

    st.markdown("---")
    tabs = st.tabs(["🔍 Preview", "📈 Class Distribution", "📉 Feature Stats", "🗂️ Column List"])

    with tabs[0]:
        st.dataframe(df_clean.head(50), use_container_width=True)

    with tabs[1]:
        class_counts = y.value_counts()
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
            colors = ["#3fb950" if c == LABEL_BENIGN else "#f85149" for c in class_counts.index]
            wedges, texts, autotexts = ax.pie(
                class_counts.values, labels=class_counts.index,
                autopct="%1.1f%%", colors=colors, startangle=90,
                textprops={"color": "#c9d1d9"},
            )
            for at in autotexts:
                at.set_color("#c9d1d9")
            ax.set_title("Class Distribution (Pie)", color="#c9d1d9")
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_bar:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
            colors = ["#3fb950" if c == LABEL_BENIGN else "#f85149" for c in class_counts.index]
            ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor="#30363d")
            for i, v in enumerate(class_counts.values):
                ax.text(i, v + class_counts.values.max() * 0.01, f"{v:,}",
                        ha="center", color="#c9d1d9", fontsize=10)
            ax.set_facecolor("none")
            ax.tick_params(colors="#c9d1d9")
            ax.set_title("Class Distribution (Bar)", color="#c9d1d9")
            ax.set_ylabel("Count", color="#c9d1d9")
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Counts table
        st.markdown("#### Class Counts")
        count_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage": [f"{v/len(y)*100:.2f}%" for v in class_counts.values],
        })
        st.dataframe(count_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("#### Descriptive Statistics (top 20 features)")
        st.dataframe(X.iloc[:, :20].describe().T.style.format("{:.3f}"), use_container_width=True)

    with tabs[3]:
        col_info = pd.DataFrame({
            "Column": df_clean.columns,
            "dtype": df_clean.dtypes.values,
            "Non-Null": df_clean.notna().sum().values,
            "Null": df_clean.isna().sum().values,
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖  Model Training":
    st.markdown("<h2>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("---")

    X = st.session_state["X"]
    y = st.session_state["y"]

    if X is None or y is None:
        st.info("📂 Please load a dataset in the **Dataset Explorer** tab first.")
        st.stop()

    st.markdown(
        f"**Dataset:** `{st.session_state['dataset_name']}`  •  "
        f"**{len(X):,} samples** · **{X.shape[1]} features**"
    )

    # Training config
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        model_name = st.selectbox("Classifier", list(SUPPORTED_MODELS.keys()), index=0)
    with col_cfg2:
        test_size = st.slider("Test set size", 0.10, 0.40, 0.25, 0.05,
                              help="Fraction of data held out for evaluation.")
    with col_cfg3:
        random_seed = st.number_input("Random seed", 0, 9999, 42, 1)

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner(f"Training {model_name}…"):
            try:
                mdl = CyberThreatModel(model_name=model_name)
                metrics = mdl.train(X, y, test_size=float(test_size), random_state=int(random_seed))
                st.session_state["model"] = mdl
                st.session_state["metrics"] = metrics
                st.session_state["feature_cols"] = mdl.feature_cols
                # Save to disk
                try:
                    saved_path = mdl.save()
                    st.success(f"✅ Model trained and saved to `{saved_path}`")
                except Exception:
                    st.success("✅ Model trained successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()

    # ── Show results ─────────────────────────────────────────────────────────
    model = st.session_state["model"]
    metrics = st.session_state["metrics"]

    if model is None or not metrics:
        st.info("👆 Configure and click **Train Model** to see evaluation results.")
        st.stop()

    st.markdown("---")
    st.markdown("### 📊 Evaluation Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
    m2.metric("🔬 Precision", f"{metrics['precision']*100:.2f}%")
    m3.metric("📡 Recall", f"{metrics['recall']*100:.2f}%")
    m4.metric("⚖️ F1-Score", f"{metrics['f1_score']*100:.2f}%")

    st.markdown("---")
    eval_tabs = st.tabs(["🔳 Confusion Matrix", "📋 Classification Report", "🌟 Feature Importance"])

    with eval_tabs[0]:
        try:
            fig = model.plot_confusion_matrix()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not render confusion matrix: {exc}")

    with eval_tabs[1]:
        le = model.label_encoder
        classes = le.classes_
        report_str = metrics.get("classification_report", "")
        st.markdown(f"```\n{report_str}\n```")
        st.caption(
            f"Classes: {list(classes)}  •  "
            f"Train samples: {metrics.get('train_samples','?')}  •  "
            f"Test samples: {metrics.get('test_samples','?')}"
        )

    with eval_tabs[2]:
        try:
            fig = model.plot_feature_importance(top_n=20)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Feature importances not available for this classifier type.")
        except Exception as exc:
            st.warning(f"Could not render feature importances: {exc}")

    st.markdown("""
<div class="warning-card">
<strong>⚠️ Interpretation Guidance</strong><br>
These metrics reflect performance on the specific dataset used. They do <strong>not</strong>
guarantee equivalent performance on real-world or unseen network traffic.
High accuracy on synthetic or small datasets should be interpreted with caution.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — THREAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍  Threat Detection":
    st.markdown("<h2>🔍 Threat Detection</h2>", unsafe_allow_html=True)
    st.markdown("---")

    model: CyberThreatModel | None = st.session_state["model"]
    feature_cols: list[str] = st.session_state["feature_cols"]

    if model is None:
        st.warning("🤖 No trained model found. Please go to **Model Training** and train a model first.")
        st.stop()

    st.markdown(
        f"**Active model:** `{model.model_name}`  •  "
        f"**Features required:** {len(feature_cols)}"
    )

    pred_file = st.file_uploader(
        "Upload a CSV of network flows to classify",
        type=["csv"],
        help="Must contain the same feature columns used during training.",
        key="predict_upload",
    )

    if pred_file is None:
        st.info("👆 Upload a CSV file to classify network flows as BENIGN or ATTACK.")
        st.stop()

    try:
        df_pred_raw = load_csv(pred_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    # Validate
    valid, reason = validate_prediction_input(df_pred_raw, feature_cols)
    if not valid:
        st.error(f"❌ {reason}")
        st.stop()

    # Clean input (no label expected)
    df_pred = df_pred_raw.copy()
    df_pred.columns = [c.strip() for c in df_pred.columns]
    numeric_cols = df_pred.select_dtypes(include=[np.number]).columns
    df_pred[numeric_cols] = df_pred[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        med = df_pred[col].median()
        df_pred[col] = df_pred[col].fillna(med if not np.isnan(med) else 0)

    X_pred = df_pred[feature_cols]

    with st.spinner("Running predictions…"):
        try:
            predictions = model.predict(X_pred)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    # Results
    df_result = df_pred_raw.copy()
    df_result["Prediction"] = predictions

    n_attack = int((predictions == LABEL_ATTACK).sum())
    n_benign = int((predictions == LABEL_BENIGN).sum())
    total = len(predictions)

    st.markdown("### 🎯 Prediction Summary")
    r1, r2, r3 = st.columns(3)
    r1.metric("📋 Total Flows", f"{total:,}")
    r2.metric("🟢 BENIGN", f"{n_benign:,}", delta=f"{n_benign/total*100:.1f}%")
    r3.metric("🔴 ATTACK", f"{n_attack:,}", delta=f"-{n_attack/total*100:.1f}%", delta_color="inverse")

    # Pie chart
    col_pie, col_tbl = st.columns([1, 2])
    with col_pie:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
        ax.pie(
            [n_benign, n_attack],
            labels=["BENIGN", "ATTACK"],
            autopct="%1.1f%%",
            colors=["#3fb950", "#f85149"],
            startangle=90,
            textprops={"color": "#c9d1d9"},
        )
        ax.set_title("Prediction Breakdown", color="#c9d1d9")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_tbl:
        st.markdown("#### Results Preview (first 100 rows)")
        st.dataframe(df_result.head(100), use_container_width=True)

    # Download
    st.markdown("---")
    csv_bytes = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Results as CSV",
        data=csv_bytes,
        file_name="threat_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

    st.markdown("""
<div class="warning-card">
<strong>⚠️ Disclaimer</strong><br>
These predictions are generated by an ML model trained on the uploaded dataset.
They are for <strong>educational demonstration only</strong> and must not be used
to take any real security actions.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️  Model Info":
    st.markdown("<h2>ℹ️ Model Information</h2>", unsafe_allow_html=True)
    st.markdown("---")

    model: CyberThreatModel | None = st.session_state["model"]
    metrics: dict = st.session_state["metrics"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🤖 Model Details")
        if model:
            st.markdown(f"""
| Property | Value |
|---|---|
| **Model Type** | {model.model_name} |
| **Status** | ✅ Trained |
| **Feature Count** | {len(model.feature_cols)} |
| **Classes** | {list(model.label_encoder.classes_)} |
| **Train Samples** | {metrics.get("train_samples", "—"):,} |
| **Test Samples** | {metrics.get("test_samples", "—"):,} |
""")
        else:
            st.info("No model has been trained yet.")

        st.markdown("### 📂 Dataset")
        if st.session_state["dataset_name"]:
            st.markdown(f"""
| Property | Value |
|---|---|
| **Source** | {st.session_state["dataset_name"]} |
| **Total Rows** | {len(st.session_state["df_clean"]):,} |
| **Features Used** | {len(st.session_state["feature_cols"])} |
""")
        else:
            st.info("No dataset loaded.")

    with col_r:
        st.markdown("### 📊 Evaluation Metrics")
        if metrics:
            st.markdown(f"""
| Metric | Value |
|---|---|
| **Accuracy** | {metrics.get("accuracy", 0)*100:.2f}% |
| **Precision** | {metrics.get("precision", 0)*100:.2f}% |
| **Recall** | {metrics.get("recall", 0)*100:.2f}% |
| **F1-Score** | {metrics.get("f1_score", 0)*100:.2f}% |
""")
        else:
            st.info("Train a model to see evaluation metrics.")

        st.markdown("### ⚠️ Limitations & Responsible Use")
        st.markdown("""
<div class="warning-card">
<strong>Important Limitations</strong><br><br>

• <strong>Not a production system</strong> — This tool is educational only.<br>
• <strong>Concept drift</strong> — Network attack patterns change over time; a model trained
  on older data will degrade in real deployments.<br>
• <strong>Synthetic data</strong> — Results on the bundled sample do not represent
  real-world accuracy.<br>
• <strong>False negatives</strong> — No model is 100% accurate; some attacks will
  be missed.<br>
• <strong>Feature dependence</strong> — The model requires the exact features it was
  trained on; different datasets may not be compatible without retraining.<br>
• <strong>No live monitoring</strong> — This system analyses static CSV files only
  and performs no live network capture or scanning.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
| Component | Technology |
|---|---|
| Language | Python 3 |
| ML Framework | scikit-learn |
| Dashboard | Streamlit |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Persistence | joblib |
| Documentation | python-docx |
| Testing | pytest / unittest |
""")
