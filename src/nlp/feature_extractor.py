import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing.text_cleaner import TextCleaner
from src.nlp.urgency_dictionary import analyze_urgency_and_suspicion

class FeatureExtractor:
    """
    Feature extraction pipeline combining:
    1. TF-IDF vectorization
    2. Sentiment & Urgency indicators (Financial, Urgency, Channel scores)
    3. Structural numerical features (Salary/Stipend, Experience, URL/Email counts)
    """

    def __init__(self, max_tfidf_features: int = 300, use_sentence_transformers: bool = True, model_dir: str = "saved_models"):
        self.text_cleaner = TextCleaner()
        self.max_tfidf_features = max_tfidf_features
        if use_sentence_transformers:
            raise ValueError("SentenceTransformer embeddings are not configured in this build; use the unified TF-IDF pipeline.")
        self.use_sentence_transformers = False
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english')
        self.is_fitted = False

    def fit_tfidf(self, corpus: List[str]):
        """Fits the TF-IDF vectorizer on cleaned text corpus."""
        cleaned_corpus = [self.text_cleaner.clean_text(doc) for doc in corpus]
        # Avoid empty vocabulary error
        if not any(cleaned_corpus):
            cleaned_corpus = ["sample text job offer software engineer developer"]
        self.tfidf_vectorizer.fit(cleaned_corpus)
        self.is_fitted = True

    def get_tfidf_features(self, corpus: List[str]) -> np.ndarray:
        """Transforms text corpus into TF-IDF matrix."""
        cleaned_corpus = [self.text_cleaner.clean_text(doc) for doc in corpus]
        if not self.is_fitted and not hasattr(self.tfidf_vectorizer, "vocabulary_"):
            self.fit_tfidf(cleaned_corpus)
        return self.tfidf_vectorizer.transform(cleaned_corpus).toarray()

    def get_semantic_embeddings(self, corpus: List[str]) -> np.ndarray:
        """Backward-compatible alias for the active TF-IDF features."""
        return self.get_tfidf_features(corpus)

    def extract_single(self, text: str, stipend: float = 0.0, exp_years: float = 0.0,
                       domain: str = "", email: str = "") -> Dict[str, Any]:
        """Extracts complete feature dictionary for a single job posting."""
        cleaned = self.text_cleaner.clean_text(text)
        meta = self.text_cleaner.extract_metadata(text)
        urgency_analysis = analyze_urgency_and_suspicion(text, domain=domain, email=email)

        return {
            "cleaned_text": cleaned,
            "url_count": meta["url_count"],
            "email_count": meta["email_count"],
            "phone_count": meta["phone_count"],
            "stipend": float(stipend),
            "exp_years": float(exp_years),
            "financial_score": urgency_analysis["financial_score"],
            "communication_score": urgency_analysis["communication_score"],
            "urgency_score": urgency_analysis["urgency_score"],
            "total_risk_points": urgency_analysis["total_risk_points"],
            "urgency_index": urgency_analysis["urgency_index"],
            "is_free_email": int(urgency_analysis["is_free_email"]),
            "is_suspicious_domain": int(urgency_analysis["is_suspicious_domain"]),
            "urgency_analysis": urgency_analysis
        }

    def prepare_feature_matrix(self, df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Builds a combined dense feature matrix (TF-IDF + Numerical/Urgency features)
        ready for ML classification and clustering models.
        """
        texts = df['description'].fillna('').astype(str).tolist()
        
        if fit or not self.is_fitted:
            self.fit_tfidf(texts)

        tfidf_feats = self.get_tfidf_features(texts)

        numerical_feats_list = []
        for _, row in df.iterrows():
            desc = str(row.get('description', ''))
            stipend = float(row.get('stipend', 0.0))
            exp_years = float(row.get('exp_years', 0.0))
            domain = str(row.get('company_domain', ''))
            email = str(row.get('contact_email', ''))

            feat_dict = self.extract_single(desc, stipend=stipend, exp_years=exp_years, domain=domain, email=email)

            num_vec = [
                feat_dict["stipend"],
                feat_dict["exp_years"],
                feat_dict["url_count"],
                feat_dict["email_count"],
                feat_dict["phone_count"],
                feat_dict["financial_score"],
                feat_dict["communication_score"],
                feat_dict["urgency_score"],
                feat_dict["total_risk_points"],
                feat_dict["urgency_index"],
                feat_dict["is_free_email"],
                feat_dict["is_suspicious_domain"]
            ]
            numerical_feats_list.append(num_vec)

        num_matrix = np.array(numerical_feats_list)
        combined_matrix = np.hstack((tfidf_feats, num_matrix))

        tfidf_names = [f"tfidf_{i}" for i in range(tfidf_feats.shape[1])]
        num_names = [
            "stipend", "exp_years", "url_count", "email_count", "phone_count",
            "financial_score", "communication_score", "urgency_score",
            "total_risk_points", "urgency_index", "is_free_email", "is_suspicious_domain"
        ]
        feature_names = tfidf_names + num_names

        return combined_matrix, feature_names

    def save_model(self):
        """Saves fitted TF-IDF vectorizer and configuration to disk."""
        filepath = os.path.join(self.model_dir, "feature_extractor.joblib")
        joblib.dump({
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "max_tfidf_features": self.max_tfidf_features,
            "is_fitted": self.is_fitted
        }, filepath)

    def load_model(self) -> bool:
        """Loads saved feature extractor model from disk."""
        filepath = os.path.join(self.model_dir, "feature_extractor.joblib")
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.tfidf_vectorizer = data["tfidf_vectorizer"]
                self.max_tfidf_features = data.get("max_tfidf_features", 300)
                self.is_fitted = data.get("is_fitted", True)
                return True
            except Exception:
                return False
        return False
