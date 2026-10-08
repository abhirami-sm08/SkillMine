from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from src.models.pipeline import build_model_pipeline


def make_evaluation_groups(df: pd.DataFrame) -> np.ndarray:
    # First group exact/repeated descriptions together. This prevents duplicated
    # template text from appearing in both train and validation folds.
    if "description" in df.columns:
        normalized = (
            df["description"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        if normalized.nunique() >= 2:
            return normalized.values

    # Optional stronger grouping for datasets that provide explicit template IDs.
    if "template_id" in df.columns and df["template_id"].notna().any():
        return df["template_id"].fillna("unknown").astype(str).values

    # Last-resort grouping by employer identity.
    for column in ("company_name", "company_domain"):
        if column in df.columns and df[column].notna().any():
            return df[column].fillna("unknown").astype(str).values

    return np.arange(len(df))


def evaluate_group_kfold(
    df: pd.DataFrame,
    label_column: str = "label",
    n_splits: int = 5,
    max_tfidf_features: int = 300,
) -> Dict[str, Any]:
    texts = df["description"].fillna("").astype(str).values
    y = df[label_column].astype(str).values
    groups = make_evaluation_groups(df)
    unique_groups = len(set(groups))
    splits = min(n_splits, unique_groups)
    if splits < 2:
        raise ValueError("GroupKFold requires at least two distinct groups.")

    fold_reports = []
    all_true = []
    all_pred = []
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(texts, y, groups):
        pipeline = build_model_pipeline(max_tfidf_features=max_tfidf_features)
        pipeline.fit(texts[train_idx], y[train_idx])
        preds = pipeline.predict(texts[test_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(preds)
        fold_reports.append(classification_report(y[test_idx], preds, output_dict=True, zero_division=0))

    labels = sorted(set(y))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for label, p, r, f in zip(labels, precision, recall, f1)
    }
    return {
        "n_splits": splits,
        "per_class": per_class,
        "fraud_metrics": per_class.get("Fraudulent", per_class.get("Fraud", {})),
        "confusion_matrix": confusion_matrix(all_true, all_pred, labels=labels).tolist(),
        "labels": labels,
        "fold_reports": fold_reports,
    }


def evaluate_classifier_performance(df: pd.DataFrame, label_column: str = "label") -> Dict[str, Any]:
    """Backward-compatible entry point used by older imports."""
    return evaluate_group_kfold(df, label_column=label_column)
