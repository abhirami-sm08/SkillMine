import os
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.text_cleaner import TextCleaner


PIPELINE_FILENAME = "model_pipeline.joblib"
CLASSES = ["Genuine", "Suspicious", "Fraudulent"]


class AdaptiveTruncatedSVD(BaseEstimator, TransformerMixin):
    """Use SVD when the feature space is large enough; otherwise pass features through."""

    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer_: Optional[TruncatedSVD] = None

    def fit(self, X, y=None):
        usable_components = min(self.n_components, max(1, min(X.shape) - 1))
        if usable_components >= 2:
            self.reducer_ = TruncatedSVD(n_components=usable_components, random_state=self.random_state)
            self.reducer_.fit(X, y)
        else:
            self.reducer_ = None
        return self

    def transform(self, X):
        if self.reducer_ is None:
            return X.toarray() if hasattr(X, "toarray") else X
        return self.reducer_.transform(X)


class CleanTextVectorizer(TfidfVectorizer):
    def __init__(self, max_features: int = 10000, ngram_range=(1, 2)):
        self.cleaner = TextCleaner()
        super().__init__(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            preprocessor=self.cleaner.clean_text,
        )


def build_model_pipeline(max_tfidf_features: int = 10000, svd_components: int = 0) -> Pipeline:
    steps = [
        ("tfidf", CleanTextVectorizer(max_features=max_tfidf_features, ngram_range=(1, 2))),
    ]
    if svd_components and svd_components > 0:
        steps.append(("svd", AdaptiveTruncatedSVD(n_components=svd_components, random_state=42)))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        )
    )
    return Pipeline(
        steps=steps
    )


def save_pipeline(pipeline: Pipeline, model_dir: str = "saved_models") -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, PIPELINE_FILENAME)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(model_dir: str = "saved_models") -> Pipeline:
    return joblib.load(os.path.join(model_dir, PIPELINE_FILENAME))


def predict_proba_by_class(pipeline: Pipeline, texts: Iterable[str]) -> List[Dict[str, float]]:
    raw_probs = pipeline.predict_proba(list(texts))
    raw_classes = list(getattr(pipeline.named_steps["classifier"], "classes_", CLASSES))
    output = []
    for row in raw_probs:
        mapped = {name: 0.0 for name in CLASSES}
        for idx, cls_name in enumerate(raw_classes):
            mapped[str(cls_name)] = float(row[idx])
        output.append(mapped)
    return output


def cap_text_only_probabilities(probabilities: Dict[str, float], secondary_verification_passed: bool = False) -> Dict[str, float]:
    """Backward-compatible helper that no longer distorts calibrated probabilities.

    The previous implementation clipped probabilities to 0.85 and then normalized them,
    which could change the actual model probabilities. The UI should display raw model
    probabilities instead of manufacturing an 85% ceiling.
    """
    if not probabilities:
        return {key: 1.0 / len(CLASSES) for key in CLASSES}
    cleaned = {key: max(0.0, float(probabilities.get(key, 0.0))) for key in CLASSES}
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 1.0 / len(CLASSES) for key in CLASSES}
    return {key: value / total for key, value in cleaned.items()}

def prediction_status(probabilities: Dict[str, float], minimum_signal: float = 0.55) -> str:
    if not probabilities or max(probabilities.values()) < minimum_signal:
        return "Unverified / Insufficient Evidence"
    return "Verified by Text Signals"
