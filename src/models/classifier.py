import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone

CLASSES = ["Genuine", "Suspicious", "Fraudulent"]
BINARY_CLASSES = ["Genuine", "Fraudulent"]


class MultiClassFraudClassifier:
    """
    Binary-first fraud classifier used by ScamShield.

    Real job-fraud datasets such as EMSCAD provide two verified labels:
    Genuine and Fraudulent. "Suspicious" is therefore an operational review
    state derived from low confidence / small probability margin at inference;
    it is not fabricated as a third training label.
    """

    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.models = {
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=2500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
            ),
        }
        self.best_model_name = "Logistic Regression"
        self.best_model = self.models[self.best_model_name]
        self.evaluation_results: Dict[str, Any] = {}
        self.fraud_threshold = 0.50
        self.training_label_mode = "binary"

    @staticmethod
    def _normalize_labels(y: np.ndarray) -> np.ndarray:
        """Map common binary labels to ScamShield's canonical labels."""
        arr = np.asarray(y, dtype=object)
        out = []
        for value in arr:
            text = str(value).strip().lower()
            if text in {"1", "1.0", "fraud", "fraudulent", "fake", "scam", "true"}:
                out.append("Fraudulent")
            elif text in {"0", "0.0", "genuine", "real", "legitimate", "false"}:
                out.append("Genuine")
            elif text == "suspicious":
                # Kept for compatibility with old three-class files.
                out.append("Suspicious")
            else:
                out.append(str(value).strip().title())
        return np.asarray(out, dtype=str)

    @staticmethod
    def _fraud_probability(model, X: np.ndarray) -> np.ndarray:
        """Return P(Fraudulent) for a binary model."""
        if not hasattr(model, "predict_proba"):
            preds = model.predict(X)
            return np.asarray([1.0 if str(v) == "Fraudulent" else 0.0 for v in preds])
        raw = model.predict_proba(X)
        classes = [str(v) for v in getattr(model, "classes_", BINARY_CLASSES)]
        if "Fraudulent" in classes:
            return raw[:, classes.index("Fraudulent")]
        if "1" in classes:
            return raw[:, classes.index("1")]
        return np.zeros(X.shape[0], dtype=float)

    @staticmethod
    def _choose_threshold(y_true: np.ndarray, fraud_probs: np.ndarray) -> float:
        """Choose a fraud threshold that maximizes fraud-class F1 on validation data."""
        candidates = np.round(np.arange(0.10, 0.91, 0.02), 2)
        best_threshold = 0.50
        best_score = -1.0
        for threshold in candidates:
            pred = np.where(fraud_probs >= threshold, "Fraudulent", "Genuine")
            score = f1_score(
                y_true,
                pred,
                labels=["Fraudulent"],
                average="binary",
                pos_label="Fraudulent",
                zero_division=0,
            )
            # Tie-break toward the less aggressive threshold.
            if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
                best_score = float(score)
                best_threshold = float(threshold)
        return best_threshold

    def _evaluate_binary(self, y_true: np.ndarray, fraud_probs: np.ndarray, threshold: float) -> Dict[str, Any]:
        pred = np.where(fraud_probs >= threshold, "Fraudulent", "Genuine")
        fraud_precision = precision_score(y_true, pred, pos_label="Fraudulent", zero_division=0)
        fraud_recall = recall_score(y_true, pred, pos_label="Fraudulent", zero_division=0)
        fraud_f1 = f1_score(y_true, pred, pos_label="Fraudulent", zero_division=0)
        try:
            roc_auc = roc_auc_score((y_true == "Fraudulent").astype(int), fraud_probs)
        except ValueError:
            roc_auc = 0.0
        try:
            pr_auc = average_precision_score((y_true == "Fraudulent").astype(int), fraud_probs)
        except ValueError:
            pr_auc = 0.0
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "precision_fraud": float(fraud_precision),
            "recall_fraud": float(fraud_recall),
            "f1_fraud": float(fraud_f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "threshold": float(threshold),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=["Genuine", "Fraudulent"]).tolist(),
            "report": classification_report(
                y_true,
                pred,
                labels=["Genuine", "Fraudulent"],
                target_names=["Genuine", "Fraudulent"],
                output_dict=True,
                zero_division=0,
            ),
        }

    @staticmethod
    def _evaluate_multiclass(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        report = classification_report(y_true, pred, output_dict=True, zero_division=0)
        fraud_row = report.get("Fraudulent", {"precision": 0, "recall": 0, "f1-score": 0})
        labels = [c for c in CLASSES if c in set(y_true)]
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "precision_fraud": float(fraud_row.get("precision", 0.0)),
            "recall_fraud": float(fraud_row.get("recall", 0.0)),
            "f1_fraud": float(fraud_row.get("f1-score", 0.0)),
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "threshold": 0.50,
            "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
            "confusion_labels": labels,
            "report": report,
        }

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        groups: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Train candidates, tune the operating threshold on an inner validation set,
        then evaluate the selected candidates on an untouched outer test set.
        """
        from sklearn.model_selection import StratifiedGroupKFold, train_test_split

        X = np.asarray(X)
        y = self._normalize_labels(y)
        if set(np.unique(y)) - set(BINARY_CLASSES):
            # If a legacy 3-class dataset is supplied, retain it rather than
            # silently relabeling Suspicious. The website still supports it.
            self.training_label_mode = "multiclass"
        else:
            self.training_label_mode = "binary"

        if X_test is None or y_test is None:
            X_train, X_eval, y_train, y_eval = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        else:
            X_train, y_train = X, y
            X_eval, y_eval = np.asarray(X_test), self._normalize_labels(y_test)

        # Inner validation for threshold selection. When groups are available,
        # duplicate/repeated descriptions stay within the same fold.
        if groups is not None and len(np.unique(groups)) >= 3:
            unique_groups = np.asarray(groups)
            cv = StratifiedGroupKFold(n_splits=min(5, len(np.unique(unique_groups))), shuffle=True, random_state=42)
            inner_train_idx, inner_val_idx = next(cv.split(X_train, y_train, unique_groups))
        else:
            inner_train_idx, inner_val_idx = train_test_split(
                np.arange(len(y_train)),
                test_size=0.20,
                random_state=42,
                stratify=y_train if len(np.unique(y_train)) > 1 else None,
            )

        results: Dict[str, Any] = {}
        candidate_models: Dict[str, Any] = {}
        best_score = -1.0

        for name, template_model in self.models.items():
            model_for_threshold = clone(template_model)
            model_for_threshold.fit(X_train[inner_train_idx], y_train[inner_train_idx])
            val_probs = self._fraud_probability(model_for_threshold, X_train[inner_val_idx])

            if self.training_label_mode == "binary":
                threshold = self._choose_threshold(y_train[inner_val_idx], val_probs)
            else:
                threshold = 0.50

            # Fit candidate on all outer training data before touching the test set.
            model = clone(template_model)
            model.fit(X_train, y_train)
            if self.training_label_mode == "binary":
                test_probs = self._fraud_probability(model, X_eval)
                metrics = self._evaluate_binary(y_eval, test_probs, threshold)
            else:
                test_preds = model.predict(X_eval)
                metrics = self._evaluate_multiclass(y_eval, test_preds)
            metrics["selection_score"] = metrics["f1_fraud"]

            results[name] = metrics
            candidate_models[name] = model

            if metrics["selection_score"] > best_score:
                best_score = metrics["selection_score"]
                self.best_model_name = name
                self.fraud_threshold = threshold

        self.best_model = candidate_models[self.best_model_name]
        self.evaluation_results = results
        self.save_model()

        return {"best_model_name": self.best_model_name, "all_results": results}

    def refit_best_model(self, X_full: np.ndarray, y_full: np.ndarray):
        """Refit the selected model on the complete labelled dataset."""
        template = self.models[self.best_model_name]
        self.best_model = clone(template)
        self.best_model.fit(np.asarray(X_full), self._normalize_labels(y_full))
        self.save_model()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return Genuine/Fraudulent and derive Suspicious from uncertainty."""
        X = np.asarray(X)
        if self.training_label_mode == "binary" and hasattr(self.best_model, "predict_proba"):
            fraud_probs = self._fraud_probability(self.best_model, X)
            genuine_probs = 1.0 - fraud_probs
            preds = []
            for pg, pf in zip(genuine_probs, fraud_probs):
                if abs(pg - pf) < 0.10 or max(pg, pf) < 0.60:
                    preds.append("Suspicious")
                else:
                    preds.append("Fraudulent" if pf >= self.fraud_threshold else "Genuine")
            return np.asarray(preds, dtype=object)
        return self.best_model.predict(X).astype(object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities aligned to Genuine/Suspicious/Fraudulent."""
        X = np.asarray(X)
        if self.training_label_mode == "binary":
            fraud_probs = np.clip(self._fraud_probability(self.best_model, X), 0.0, 1.0)
            genuine_probs = 1.0 - fraud_probs
            suspicious_probs = np.zeros_like(fraud_probs)
            return np.column_stack([genuine_probs, suspicious_probs, fraud_probs])

        if hasattr(self.best_model, "predict_proba"):
            raw_probs = self.best_model.predict_proba(X)
            raw_classes = [str(v) for v in getattr(self.best_model, "classes_", CLASSES)]
            full_probs = np.zeros((X.shape[0], len(CLASSES)))
            for idx, cls_name in enumerate(CLASSES):
                if cls_name in raw_classes:
                    full_probs[:, idx] = raw_probs[:, raw_classes.index(cls_name)]
            return full_probs

        preds = self.best_model.predict(X)
        probs = np.zeros((len(preds), len(CLASSES)))
        for i, p in enumerate(preds):
            if p in CLASSES:
                probs[i, CLASSES.index(p)] = 1.0
        return probs

    def get_feature_importances(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Return top feature weights/importances for the selected model."""
        if hasattr(self.best_model, "feature_importances_"):
            importances = np.asarray(self.best_model.feature_importances_)
        elif hasattr(self.best_model, "coef_"):
            coef = np.asarray(self.best_model.coef_)
            importances = np.abs(coef[0]) if coef.ndim == 2 else np.abs(coef)
        else:
            return []

        if len(importances) != len(feature_names):
            return []
        indices = np.argsort(importances)[::-1]
        return [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
            for i in indices[:20]
        ]

    def save_model(self):
        filepath = os.path.join(self.model_dir, "best_classifier.joblib")
        joblib.dump(
            {
                "best_model_name": self.best_model_name,
                "best_model": self.best_model,
                "evaluation_results": self.evaluation_results,
                "classes": CLASSES,
                "fraud_threshold": self.fraud_threshold,
                "training_label_mode": self.training_label_mode,
            },
            filepath,
        )

    def load_model(self) -> bool:
        filepath = os.path.join(self.model_dir, "best_classifier.joblib")
        if not os.path.exists(filepath):
            return False
        try:
            data = joblib.load(filepath)
            self.best_model_name = data["best_model_name"]
            self.best_model = data["best_model"]
            self.evaluation_results = data.get("evaluation_results", {})
            self.fraud_threshold = float(data.get("fraud_threshold", 0.50))
            self.training_label_mode = data.get("training_label_mode", "binary")
            return True
        except Exception:
            return False
