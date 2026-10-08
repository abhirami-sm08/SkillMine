import sys
import os
import json
import pandas as pd
import numpy as np
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from sklearn.model_selection import StratifiedGroupKFold

from src.nlp.feature_extractor import FeatureExtractor
from src.nlp.urgency_dictionary import analyze_urgency_and_suspicion
from src.models.classifier import MultiClassFraudClassifier
from src.models.clustering import ClusterAndAnomalyEngine
from src.models.pipeline import build_model_pipeline, save_pipeline
from src.association_rules.rule_miner import AssociationRuleMiner
from src.trust_engine.scoring import TrustScoreEngine
from src.rules import find_blacklist_matches
from database.db_config import init_db, SessionLocal
from database.models import ScannedOffer, BlacklistEntity, AnalyticsLog


def normalized_description_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )


def load_training_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    if "label" not in df.columns and "fraudulent" in df.columns:
        fraud = pd.to_numeric(df["fraudulent"], errors="coerce").fillna(0).astype(int)
        df["label"] = np.where(fraud.eq(1), "Fraudulent", "Genuine")

    if "label" not in df.columns:
        raise ValueError("Dataset must contain `label` or `fraudulent`.")

    if "description" not in df.columns:
        text_parts = [
            c for c in ["title", "company_profile", "description", "requirements", "benefits"]
            if c in df.columns
        ]
        if not text_parts:
            raise ValueError("Dataset must contain job-posting text columns.")
        df["description"] = df[text_parts].fillna("").astype(str).agg(" ".join, axis=1)

    required_defaults = {
        "title": "Untitled Offer",
        "company_name": "Not provided",
        "company_domain": "",
        "contact_email": "",
        "contact_phone": "",
        "department": "General",
        "location": "Remote / Not provided",
        "stipend": 0.0,
        "exp_years": 0.0,
        "posting_date": pd.NaT,
        "source": "custom",
    }
    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    # Ensure numeric fields do not contain NaN/strings.
    df["stipend"] = pd.to_numeric(df["stipend"], errors="coerce").fillna(0.0)
    df["exp_years"] = pd.to_numeric(df["exp_years"], errors="coerce").fillna(0.0)
    df["label"] = df["label"].astype(str).str.strip().str.title()
    df = df[df["label"].isin(["Genuine", "Fraudulent", "Suspicious"])].copy()

    # Exact duplicate rows are safe to remove. For real-data training, repeated
    # descriptions are retained but kept in the same grouped validation split.
    df = df.drop_duplicates().reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("Training requires at least two label classes.")
    return df


def run_training_pipeline(csv_path: str = None):
    print("==============================================================")
    print("SCAMSHIELD AI — REAL LABELLED DATA TRAINING PIPELINE")
    print("==============================================================")

    data_dir = os.path.join(ROOT_DIR, "data")
    if csv_path is None:
        csv_path = os.path.join(data_dir, "job_offers.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training dataset not found: {csv_path}\n"
            "Run `python scripts/setup_real_data.py` first."
        )

    print(f"[*] Training data source: {csv_path}")
    df = load_training_data(csv_path)

    norm_desc = normalized_description_series(df["description"])
    desc_unique = int(norm_desc.nunique())
    dup_ratio = float(1.0 - desc_unique / max(1, len(df)))
    print(f"[*] Loaded {len(df):,} job offers.")
    print(f"[*] Unique normalized descriptions: {desc_unique:,} ({dup_ratio*100:.2f}% duplication)")
    print("[*] Label distribution:")
    print(df["label"].value_counts())

    # Outer evaluation split: grouped by normalized description to prevent
    # duplicated/template ads crossing train/test.
    labels = df["label"].values
    outer_groups = norm_desc.values
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(outer_cv.split(df, labels, groups=outer_groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"[*] Grouped outer split: train={len(df_train):,}, test={len(df_test):,}")

    print("[1/5] Fitting evaluation TF-IDF only on the outer training split...")
    eval_extractor = FeatureExtractor(max_tfidf_features=300, use_sentence_transformers=False,
                                      model_dir=os.path.join(ROOT_DIR, "saved_models"))
    X_train, _ = eval_extractor.prepare_feature_matrix(df_train, fit=True)
    X_test, _ = eval_extractor.prepare_feature_matrix(df_test, fit=False)

    print("[2/5] Training and comparing fraud classifiers...")
    groups_train = normalized_description_series(df_train["description"]).values
    classifier = MultiClassFraudClassifier(model_dir=os.path.join(ROOT_DIR, "saved_models"))
    eval_results = classifier.train_and_evaluate(
        X_train,
        df_train["label"].values,
        X_test=X_test,
        y_test=df_test["label"].values,
        groups=groups_train,
    )

    print(f"[+] Best model: {eval_results['best_model_name']}")
    print(f"[+] Fraud decision threshold: {classifier.fraud_threshold:.2f}")
    for name, metrics in eval_results["all_results"].items():
        print(
            f"   - {name:<20} Accuracy {metrics['accuracy']*100:6.2f}% | "
            f"Fraud Precision {metrics['precision_fraud']*100:6.2f}% | "
            f"Fraud Recall {metrics['recall_fraud']*100:6.2f}% | "
            f"Fraud F1 {metrics['f1_fraud']:.3f} | PR-AUC {metrics['pr_auc']:.3f}"
        )

    print("[3/5] Refitting the selected classifier on the complete labelled dataset...")
    feature_extractor = FeatureExtractor(max_tfidf_features=300, use_sentence_transformers=False,
                                         model_dir=os.path.join(ROOT_DIR, "saved_models"))
    X_full, feature_names = feature_extractor.prepare_feature_matrix(df, fit=True)
    feature_extractor.save_model()
    classifier.refit_best_model(X_full, df["label"].values)

    # Store feature names used by the classifier so the performance center can
    # show meaningful weights/importances after retraining.
    feat_path = os.path.join(ROOT_DIR, "saved_models", "feature_names.json")
    with open(feat_path, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh)

    print("[4/5] Fitting clustering, anomaly detection and raw-text compatibility model...")
    num_submatrix = X_full[:, -12:]
    clustering_engine = ClusterAndAnomalyEngine(model_dir=os.path.join(ROOT_DIR, "saved_models"), n_clusters=4)
    clustering_engine.fit(
        X_full,
        num_submatrix,
        texts=df["description"].fillna("").astype(str).tolist(),
    )
    clusters = clustering_engine.predict_cluster(X_full)
    anomalies, anomaly_scores = clustering_engine.detect_anomaly(num_submatrix)

    raw_pipeline = build_model_pipeline(max_tfidf_features=10000)
    raw_pipeline.fit(df["description"].fillna("").astype(str).tolist(), df["label"].astype(str).values)
    save_pipeline(raw_pipeline, model_dir=os.path.join(ROOT_DIR, "saved_models"))

    print("[5/5] Updating association-rule mining + local analytics database...")
    rule_miner = AssociationRuleMiner(min_support=0.02, min_confidence=0.50, min_lift=1.20)
    try:
        mined_rules = rule_miner.mine_rules(df)
        print(f"   - Mined {len(mined_rules)} association rules.")
    except RuntimeError as exc:
        # The classifier must remain usable even when optional analytics dependency
        # mlxtend has not been installed yet. `pip install -r requirements.txt`
        # enables this feature for the final project.
        mined_rules = pd.DataFrame()
        print(f"   - WARNING: Association-rule mining skipped: {exc}")

    init_db()
    session = SessionLocal()
    blacklist_records = session.query(BlacklistEntity).all()
    blacklist_dicts = [
        {"type": bl.entity_type, "value": bl.entity_value, "reason": bl.reason}
        for bl in blacklist_records
    ]
    trust_engine = TrustScoreEngine()
    session.query(ScannedOffer).delete()

    scanned = []
    for idx, row in df.iterrows():
        x = X_full[idx:idx + 1]
        pred = str(classifier.predict(x)[0])
        probs = classifier.predict_proba(x)[0]
        p_dict = {c: float(probs[i]) for i, c in enumerate(["Genuine", "Suspicious", "Fraudulent"])}
        desc = str(row.get("description", ""))
        domain = str(row.get("company_domain", ""))
        email = str(row.get("contact_email", ""))
        urgency = analyze_urgency_and_suspicion(desc, domain=domain, email=email)
        matches = find_blacklist_matches(
            text=desc,
            domain=domain,
            email=email,
            company_name=str(row.get("company_name", "")),
            phone=str(row.get("contact_phone", "")),
            blacklist_entries=blacklist_dicts,
        )
        scoring = trust_engine.compute_trust_score(
            predicted_class=pred,
            class_probabilities=p_dict,
            urgency_analysis=urgency,
            is_anomaly=bool(anomalies[idx]),
            anomaly_score=float(anomaly_scores[idx]),
            blacklist_matches=matches,
            raw_text=desc,
        )
        posting_date = pd.to_datetime(row.get("posting_date"), errors="coerce")
        scanned.append(
            ScannedOffer(
                title=str(row.get("title", "Untitled Offer")),
                company_name=str(row.get("company_name", "Not provided")),
                company_domain=domain,
                contact_email=email,
                contact_phone=str(row.get("contact_phone", "")),
                department=str(row.get("department", "General")),
                location=str(row.get("location", "Remote / Not provided")),
                stipend=float(row.get("stipend", 0.0)),
                exp_years=float(row.get("exp_years", 0.0)),
                description=desc,
                posting_date=posting_date.to_pydatetime() if pd.notna(posting_date) else None,
                predicted_class=pred,
                trust_score=float(scoring["trust_score"]),
                archetype_cluster=int(clusters[idx]),
                is_anomaly=bool(anomalies[idx]),
                anomaly_score=float(anomaly_scores[idx]),
                urgency_index=float(urgency.get("urgency_index", 0.0)),
                risk_factors_json=json.dumps(scoring["explanation_alerts"]),
            )
        )

    session.add_all(scanned)
    session.add(
        AnalyticsLog(
            event_type="MODEL_TRAINED",
            details=f"Trained {eval_results['best_model_name']} on {len(df):,} labelled records from {os.path.basename(csv_path)}.",
        )
    )
    session.commit()
    session.close()

    metadata = {
        "dataset_path": csv_path,
        "dataset_source": str(df["source"].iloc[0]) if "source" in df.columns else "custom",
        "rows": int(len(df)),
        "unique_descriptions": desc_unique,
        "description_duplication": dup_ratio,
        "labels": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "outer_test_rows": int(len(df_test)),
        "best_model": eval_results["best_model_name"],
        "fraud_threshold": float(classifier.fraud_threshold),
        "evaluation": eval_results["all_results"],
        "note": "Accuracy is secondary; fraud precision/recall/F1 and PR-AUC are the primary monitoring metrics.",
    }
    with open(os.path.join(ROOT_DIR, "saved_models", "training_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("==============================================================")
    print("[OK] REAL-DATA MODEL TRAINING COMPLETED")
    print("==============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ScamShield on labelled job-posting data.")
    parser.add_argument("--data", default=os.path.join(ROOT_DIR, "data", "job_offers.csv"),
                        help="Prepared ScamShield CSV. Default: data/job_offers.csv")
    args = parser.parse_args()
    run_training_pipeline(args.data)
