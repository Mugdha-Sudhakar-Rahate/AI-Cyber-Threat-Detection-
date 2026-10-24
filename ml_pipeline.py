"""
ml_pipeline.py
--------------
Machine-learning pipeline for binary cyber-threat classification.

Wraps a scikit-learn Pipeline (StandardScaler → RandomForestClassifier) and
exposes a clean interface used by both the Streamlit dashboard and the Jupyter
Notebook.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in all contexts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "threat_model.joblib")

SUPPORTED_MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
}


# ─────────────────────────────────────────────────────────────────────────────
# CyberThreatModel
# ─────────────────────────────────────────────────────────────────────────────

class CyberThreatModel:
    """
    Wraps a scikit-learn classification pipeline for cyber-threat detection.

    Attributes
    ----------
    model_name : str
    pipeline   : sklearn.pipeline.Pipeline  (set after training)
    feature_cols : list[str]                (set after training)
    label_encoder : LabelEncoder            (set after training)
    metrics    : dict                       (set after training)
    is_trained : bool
    """

    def __init__(self, model_name: str = "Random Forest"):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model_name
        self.pipeline: Pipeline | None = None
        self.feature_cols: list[str] = []
        self.label_encoder = LabelEncoder()
        self.metrics: dict = {}
        self.is_trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> dict:
        """
        Train the pipeline on (X, y) and evaluate on a held-out test split.

        Parameters
        ----------
        X            : feature DataFrame (numeric columns only)
        y            : label Series (BENIGN / ATTACK strings)
        test_size    : fraction reserved for testing
        random_state : seed for reproducibility

        Returns
        -------
        dict of evaluation metrics
        """
        if len(X) < 10:
            raise ValueError(
                "Dataset too small to train: need at least 10 rows."
            )

        unique_classes = y.nunique()
        if unique_classes < 2:
            raise ValueError(
                "Only one class found in the labels. "
                "Provide a dataset with both BENIGN and ATTACK examples."
            )

        self.feature_cols = list(X.columns)

        # Encode labels
        y_enc = self.label_encoder.fit_transform(y)

        # Stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=test_size, random_state=random_state,
                stratify=y_enc,
            )
        except ValueError:
            # Fallback if stratification fails (very small dataset)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=test_size, random_state=random_state,
            )

        # Build pipeline
        clf = SUPPORTED_MODELS[self.model_name]
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        self.metrics = self._compute_metrics(y_test, y_pred)
        self.metrics["train_samples"] = len(X_train)
        self.metrics["test_samples"] = len(X_test)
        self.metrics["feature_count"] = len(self.feature_cols)

        self.is_trained = True
        return self.metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return label strings (BENIGN / ATTACK) for each row in X.
        """
        self._check_trained()
        X_aligned = self._align_features(X)
        y_enc = self.pipeline.predict(X_aligned)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return probability matrix if the classifier supports it, else None."""
        self._check_trained()
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "predict_proba"):
            X_aligned = self._align_features(X)
            return self.pipeline.predict_proba(X_aligned)
        return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH) -> str:
        """Persist the trained model to disk using joblib."""
        self._check_trained()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "pipeline": self.pipeline,
            "feature_cols": self.feature_cols,
            "label_encoder": self.label_encoder,
            "metrics": self.metrics,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "CyberThreatModel":
        """Load a previously saved model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at: {path}")
        payload = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model_name = payload["model_name"]
        obj.pipeline = payload["pipeline"]
        obj.feature_cols = payload["feature_cols"]
        obj.label_encoder = payload["label_encoder"]
        obj.metrics = payload["metrics"]
        obj.is_trained = True
        return obj

    # ── Visualisations ────────────────────────────────────────────────────────

    def plot_confusion_matrix(self) -> plt.Figure:
        """Return a Matplotlib Figure of the confusion matrix."""
        self._check_trained()
        cm = self.metrics.get("confusion_matrix")
        if cm is None:
            raise ValueError("No confusion matrix available. Train the model first.")

        classes = self.label_encoder.classes_
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_feature_importance(self, top_n: int = 20) -> plt.Figure | None:
        """
        Return a Matplotlib Figure of top-N feature importances.
        Returns None if the classifier does not support feature_importances_.
        """
        self._check_trained()
        clf = self.pipeline.named_steps["classifier"]
        if not hasattr(clf, "feature_importances_"):
            return None

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [self.feature_cols[i] for i in indices]
        top_values = importances[indices]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))  # type: ignore[attr-defined]
        ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(f"Top-{top_n} Feature Importances", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_class_distribution(self, y: pd.Series) -> plt.Figure:
        """Return a Matplotlib Figure showing class distribution."""
        counts = y.value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            counts.index, counts.values,
            color=["#2196F3" if c == "BENIGN" else "#F44336" for c in counts.index],
            edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.values.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10,
            )
        ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlabel("Class", fontsize=11)
        fig.tight_layout()
        return fig

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_trained(self):
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError(
                "Model has not been trained yet. Call train() first."
            )

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align X columns to the training feature set.
        Missing columns are filled with 0; extra columns are dropped.
        """
        missing = set(self.feature_cols) - set(X.columns)
        for col in missing:
            X = X.copy()
            X[col] = 0.0
        return X[self.feature_cols]

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute a comprehensive set of classification metrics."""
        avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
        pos_label = 1 if avg == "binary" else None

        kwargs = {"average": avg, "zero_division": 0}
        if avg == "binary":
            kwargs["pos_label"] = pos_label  # type: ignore[assignment]

        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, **kwargs), 4),
            "recall": round(recall_score(y_true, y_pred, **kwargs), 4),
            "f1_score": round(f1_score(y_true, y_pred, **kwargs), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0
            ),
        }
