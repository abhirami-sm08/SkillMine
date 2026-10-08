import sys
import os
import unittest
import shutil
import pandas as pd
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_cleaner import TextCleaner
from src.nlp.urgency_dictionary import analyze_urgency_and_suspicion
from src.nlp.feature_extractor import FeatureExtractor
from src.models.classifier import MultiClassFraudClassifier
from src.models.clustering import ClusterAndAnomalyEngine
from src.models.pipeline import build_model_pipeline, save_pipeline, load_pipeline, predict_proba_by_class
from src.association_rules.rule_miner import AssociationRuleMiner
from src.trust_engine.scoring import TrustScoreEngine
from src.rules import detect_payment_demand, find_blacklist_matches
from src.domain_intelligence import detect_lookalike_domain, recruiter_domain_matches
from database.db_config import init_db, SessionLocal
from database.models import ScannedOffer, BlacklistEntity
from database.olap_engine import OLAPAnalyticsEngine

class TestScamShieldPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.tmpdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".codex_tmp", "test_models")
        shutil.rmtree(cls.tmpdir, ignore_errors=True)
        os.makedirs(cls.tmpdir, exist_ok=True)
        cls.cleaner = TextCleaner()
        cls.feat_extractor = FeatureExtractor(max_tfidf_features=100, use_sentence_transformers=False, model_dir=cls.tmpdir)
        cls.classifier = MultiClassFraudClassifier(model_dir=cls.tmpdir)
        cls.clustering = ClusterAndAnomalyEngine(model_dir=cls.tmpdir)
        cls.rule_miner = AssociationRuleMiner()
        cls.trust_engine = TrustScoreEngine()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_text_cleaner(self):
        text = "Check out http://scam-jobs.com or wa.me/919876543210 for urgent hiring! Contact hr@gmail.com!"
        cleaned = self.cleaner.clean_text(text)
        meta = self.text_cleaner_meta = self.cleaner.extract_metadata(text)
        
        self.assertNotIn("http", cleaned)
        self.assertEqual(meta["url_count"], 2) # http link and wa.me link
        self.assertEqual(meta["email_count"], 1)

    def test_02_urgency_analysis(self):
        text = "Direct selection! Pay registration fee of Rs 2500 for laptop deposit. WhatsApp only."
        res = analyze_urgency_and_suspicion(text, domain="scam.wixsite.com", email="test@gmail.com")
        
        self.assertGreater(res["financial_score"], 0)
        self.assertGreater(res["urgency_index"], 0.3)
        self.assertTrue(res["is_free_email"])
        self.assertTrue(res["is_suspicious_domain"])

    def test_03_synthetic_dataset_and_training(self):
        from scripts.seed_data import generate_synthetic_dataset
        df = generate_synthetic_dataset(num_records=50)
        
        X_mat, feat_names = self.feat_extractor.prepare_feature_matrix(df, fit=True)
        self.assertEqual(X_mat.shape[0], 50)

        # Train Classifier
        y_labels = df['label'].values
        res = self.classifier.train_and_evaluate(X_mat, y_labels)
        self.assertIn("best_model_name", res)

        # Train Clustering & Anomaly
        num_sub = X_mat[:, -12:]
        self.clustering.fit(X_mat, num_sub, texts=df["description"].astype(str).tolist())
        clusters = self.clustering.predict_cluster(X_mat)
        self.assertEqual(len(clusters), 50)

        # Rule Miner
        rules = self.rule_miner.mine_rules(df)
        self.assertIsNotNone(rules)

    def test_04_trust_score_engine(self):
        urgency_res = {
            "financial_score": 10,
            "financial_matches": ["registration fee"],
            "communication_score": 8,
            "communication_matches": ["whatsapp group"],
            "urgency_score": 5,
            "urgency_matches": ["no interview"],
            "urgency_index": 0.9,
            "is_free_email": True,
            "is_suspicious_domain": True
        }

        res = self.trust_engine.compute_trust_score(
            predicted_class="Fraudulent",
            class_probabilities={"Genuine": 0.05, "Suspicious": 0.15, "Fraudulent": 0.80},
            urgency_analysis=urgency_res,
            is_anomaly=True,
            anomaly_score=0.85,
            blacklist_matches=[{"type": "DOMAIN", "value": "scam.com", "reason": "Known scam"}]
        )

        self.assertLess(res["trust_score"], 40.0)
        self.assertEqual(res["risk_level"], "HIGH RISK / FRAUDULENT")
        self.assertGreater(len(res["explanation_alerts"]), 0)

    def test_05_olap_engine(self):
        olap = OLAPAnalyticsEngine()
        df_offers = olap.load_offers_dataframe()
        self.assertIsNotNone(df_offers)

    def test_06_feature_extractor_persistence(self):
        feat_ext = FeatureExtractor(max_tfidf_features=50, use_sentence_transformers=False, model_dir=self.tmpdir)
        sample_df = pd.DataFrame([{"description": "Python Software Engineer Intern REST API SQL B.Tech"}])
        feat_ext.prepare_feature_matrix(sample_df, fit=True)
        feat_ext.save_model()

        new_feat_ext = FeatureExtractor(max_tfidf_features=50, use_sentence_transformers=False, model_dir=self.tmpdir)
        loaded = new_feat_ext.load_model()
        self.assertTrue(loaded)
        self.assertTrue(new_feat_ext.is_fitted)

        # Test prediction without fit=True
        matrix, names = new_feat_ext.prepare_feature_matrix(sample_df, fit=False)
        self.assertEqual(matrix.shape[0], 1)

    def test_07_unified_pipeline_accepts_raw_text(self):
        df = pd.DataFrame(
            [
                {"description": "Official software intern with interview and coding test", "label": "Genuine"},
                {"description": "Pay Rs 2500 registration fee to UPI fastpay@upi", "label": "Fraudulent"},
                {"description": "Telegram only urgent direct selection", "label": "Suspicious"},
                {"description": "Python backend role with assessment and HR interview", "label": "Genuine"},
                {"description": "Processing fee Rs 3500 required before joining", "label": "Fraudulent"},
                {"description": "WhatsApp only, no interview, immediate joiner", "label": "Suspicious"},
            ]
        )
        pipeline = build_model_pipeline(max_tfidf_features=50, svd_components=10)
        pipeline.fit(df["description"].tolist(), df["label"].tolist())
        path = save_pipeline(pipeline, model_dir=self.tmpdir)
        self.assertTrue(os.path.exists(path))
        loaded = load_pipeline(model_dir=self.tmpdir)
        pred = loaded.predict(["Pay Rs 2000 registration fee today"])[0]
        probs = predict_proba_by_class(loaded, ["Pay Rs 2000 registration fee today"])[0]
        self.assertIn(pred, {"Genuine", "Suspicious", "Fraudulent"})
        self.assertIn("Fraudulent", probs)

    def test_08_payment_rule_negation_and_blacklist_phone(self):
        self.assertTrue(detect_payment_demand("Pay Rs 2500 registration fee to fastpay@upi").triggered)
        self.assertFalse(detect_payment_demand("No registration fee required for this internship.").triggered)
        matches = find_blacklist_matches(
            text="Contact on WhatsApp +91-9876543210",
            blacklist_entries=[{"type": "PHONE", "value": "+91-9876543210", "reason": "reported"}],
        )
        self.assertEqual(len(matches), 1)

    def test_09_domain_intelligence_helpers(self):
        self.assertEqual(detect_lookalike_domain("infosys-careers-hr.com"), "infosys.com")
        self.assertTrue(recruiter_domain_matches("hr@infosys.com", "infosys.com"))


if __name__ == "__main__":
    unittest.main()
