"""Run a few unseen examples against the trained ScamShield model."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.nlp.feature_extractor import FeatureExtractor
from src.models.classifier import MultiClassFraudClassifier

EXAMPLES = [
    {
        "name": "Professional internship",
        "text": "Software engineering internship. Candidates complete an assessment and technical interview. No application or registration fee. Apply through the official careers portal.",
        "stipend": 18000,
        "exp_years": 0,
        "email": "careers@company.com",
        "domain": "company.com",
    },
    {
        "name": "Upfront payment scam",
        "text": "Congratulations! You are selected immediately without interview. Pay a 2500 registration and security fee today to receive the joining letter and laptop. Contact our recruiter on WhatsApp.",
        "stipend": 60000,
        "exp_years": 0,
        "email": "hr.recruitment@gmail.com",
        "domain": "",
    },
    {
        "name": "Urgent messaging offer",
        "text": "Urgent hiring for remote work. Limited seats. Message our Telegram recruiter now. No interview required. Training charges may apply after selection.",
        "stipend": 45000,
        "exp_years": 0,
        "email": "",
        "domain": "",
    },
]


def main():
    model_dir = os.path.join(ROOT, "saved_models")
    fx = FeatureExtractor(max_tfidf_features=300, use_sentence_transformers=False, model_dir=model_dir)
    classifier = MultiClassFraudClassifier(model_dir=model_dir)
    if not fx.load_model() or not classifier.load_model():
        raise RuntimeError("Train the real-data model first: python scripts/train_models.py")

    df = pd.DataFrame([
        {
            "description": e["text"],
            "stipend": e["stipend"],
            "exp_years": e["exp_years"],
            "company_domain": e["domain"],
            "contact_email": e["email"],
        }
        for e in EXAMPLES
    ])
    X, _ = fx.prepare_feature_matrix(df, fit=False)
    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)
    for i, e in enumerate(EXAMPLES):
        print(f"{e['name']}: {preds[i]} | genuine={probs[i,0]:.3f} | fraud={probs[i,2]:.3f}")


if __name__ == "__main__":
    main()
