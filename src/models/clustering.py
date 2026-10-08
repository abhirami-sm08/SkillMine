import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

ARCHETYPE_LABELS = {}

class ClusterAndAnomalyEngine:
    """
    Clustering & Outlier Anomaly Detection Engine for ScamShield.
    1. K-Means: Groups job postings into 4 fraud archetype clusters.
    2. Isolation Forest: Detects salary, stipend, and experience requirement anomalies.
    3. PCA: Reduces high-dimensional features to 2D/3D for visualization.
    """

    def __init__(self, n_clusters: int = 4, model_dir: str = "saved_models"):
        self.n_clusters = n_clusters
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.cluster_labels = {}

    def fit(self, X: np.ndarray, numerical_features: np.ndarray, texts: List[str] = None):
        """
        Fits K-Means on full combined features,
        and Isolation Forest on numerical salary/experience/urgency features.
        """
        # Fit K-Means on full feature matrix
        self.kmeans.fit(X)
        
        # Fit Isolation Forest on numerical features [stipend, exp_years, urgency_index, etc.]
        self.isolation_forest.fit(numerical_features)

        # Fit PCA transformers
        if X.shape[0] >= 3:
            self.pca_2d.fit(X)
            self.pca_3d.fit(X)

        if texts is not None:
            clusters = self.kmeans.predict(X)
            self.cluster_labels = self._derive_cluster_labels(texts, clusters)
        else:
            self.cluster_labels = {idx: f"Cluster #{idx}" for idx in range(self.n_clusters)}

        self.save_models()

    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """Assigns cluster ID (0 to n_clusters-1)."""
        return self.kmeans.predict(X)

    def get_cluster_name(self, cluster_id: int) -> str:
        """Returns human readable cluster archetype label."""
        return self.cluster_labels.get(cluster_id, ARCHETYPE_LABELS.get(cluster_id, f"Cluster #{cluster_id}"))

    def _derive_cluster_labels(self, texts: List[str], clusters: np.ndarray) -> Dict[int, str]:
        keyword_topics = {
            "Payment Demand Scam": ["registration fee", "processing fee", "deposit", "upi", "pay"],
            "Social Messaging Lead": ["telegram", "whatsapp", "dm", "group"],
            "No-Interview Pressure": ["no interview", "direct selection", "guaranteed", "urgent"],
            "Professional Corporate Posting": ["interview", "assessment", "python", "team", "requirements"],
        }
        labels = {}
        for cluster_id in range(self.n_clusters):
            cluster_text = " ".join(str(texts[i]).lower() for i, c in enumerate(clusters) if c == cluster_id)
            if not cluster_text:
                labels[cluster_id] = f"Cluster #{cluster_id}"
                continue
            scores = {
                topic: sum(cluster_text.count(keyword) for keyword in keywords)
                for topic, keywords in keyword_topics.items()
            }
            best_topic, best_score = max(scores.items(), key=lambda item: item[1])
            labels[cluster_id] = best_topic if best_score > 0 else f"Cluster #{cluster_id}"
        return labels

    def detect_anomaly(self, numerical_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Isolation Forest on numerical features.
        Returns:
            - is_anomaly: Array of booleans (True if outlier)
            - anomaly_score: Normalized 0.0 to 1.0 anomaly severity score (higher = more anomalous)
        """
        # score_samples returns raw anomaly score (lower means more anomalous)
        raw_scores = self.isolation_forest.score_samples(numerical_features)
        preds = self.isolation_forest.predict(numerical_features) # -1 for anomaly, 1 for inlier

        is_anomaly = preds == -1
        # Normalize raw scores to 0.0 - 1.0 scale (0.0 normal, 1.0 highly anomalous)
        # raw_scores typically range from -0.8 (anomaly) to -0.3 (inlier)
        min_s, max_s = -0.85, -0.35
        normalized_scores = np.clip((max_s - raw_scores) / (max_s - min_s), 0.0, 1.0)

        return is_anomaly, normalized_scores

    def transform_pca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms feature matrix X into 2D and 3D PCA coordinates for plotting."""
        coords_2d = self.pca_2d.transform(X) if hasattr(self.pca_2d, "components_") else np.zeros((X.shape[0], 2))
        coords_3d = self.pca_3d.transform(X) if hasattr(self.pca_3d, "components_") else np.zeros((X.shape[0], 3))
        return coords_2d, coords_3d

    def save_models(self):
        """Saves clustering and anomaly models to disk."""
        filepath = os.path.join(self.model_dir, "clustering_anomaly.joblib")
        joblib.dump({
            "kmeans": self.kmeans,
            "isolation_forest": self.isolation_forest,
            "pca_2d": self.pca_2d,
            "pca_3d": self.pca_3d,
            "n_clusters": self.n_clusters,
            "cluster_labels": self.cluster_labels
        }, filepath)

    def load_models(self) -> bool:
        """Loads saved models from disk."""
        filepath = os.path.join(self.model_dir, "clustering_anomaly.joblib")
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.kmeans = data["kmeans"]
                self.isolation_forest = data["isolation_forest"]
                self.pca_2d = data["pca_2d"]
                self.pca_3d = data["pca_3d"]
                self.n_clusters = data.get("n_clusters", 4)
                self.cluster_labels = data.get("cluster_labels", {})
                return True
            except Exception:
                return False
        return False
