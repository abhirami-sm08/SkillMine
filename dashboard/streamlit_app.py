import sys
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
REVIEW_QUEUE_PATH = os.path.join(ROOT_DIR, "data", "review_queue.json")

from src.nlp.feature_extractor import FeatureExtractor
from src.nlp.urgency_dictionary import analyze_urgency_and_suspicion
from src.models.classifier import MultiClassFraudClassifier
from src.models.clustering import ClusterAndAnomalyEngine
from src.models.pipeline import load_pipeline, predict_proba_by_class, cap_text_only_probabilities
from src.models.model_evaluator import evaluate_group_kfold
from src.association_rules.rule_miner import AssociationRuleMiner
from src.trust_engine.scoring import TrustScoreEngine
from src.rules import find_blacklist_matches
from src.ocr_parser import extract_text_from_upload
from src.feedback import append_review_item, load_review_queue
from database.db_config import init_db, SessionLocal
from database.models import ScannedOffer, BlacklistEntity, AnalyticsLog
from database.olap_engine import OLAPAnalyticsEngine

# Set Page Config
st.set_page_config(
    page_title="ScamShield AI | Aerospike Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aerospike Enterprise Dark Theme CSS
AEROSPIKE_CSS = """
<style>
    /* Aerospike Main Canvas: Dark Obsidian & Deep Blue */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #0b0f19 70%, #07090e 100%);
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Headings & Text */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #f8fafc !important;
    }
    
    /* Glowing Aerospike Top Banner */
    .aerospike-banner {
        background: linear-gradient(90deg, #0b0f19 0%, #111e38 50%, #0b0f19 100%);
        border: 1px solid #1e2d4a;
        border-left: 4px solid #0066FF;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 25px rgba(0, 102, 255, 0.15);
    }
    .aerospike-pill {
        background: rgba(0, 102, 255, 0.15);
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-right: 8px;
    }
    .aerospike-pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    /* Input fields styling */
    input, textarea, select, div[data-baseweb="select"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #ffffff !important;
        background-color: #111827 !important;
    }

    /* Glassmorphic Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid #1e293b;
        border-top: 2px solid #0066FF;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #0066FF !important;
    }

    /* Dataframe Table styling */
    div[data-testid="stDataFrame"] {
        background-color: #0d1322 !important;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }

    /* Custom Badges */
    .metric-badge-red {
        background: #dc2626;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .metric-badge-yellow {
        background: #d97706;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .metric-badge-green {
        background: #16a34a;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0066FF 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 20px rgba(0, 102, 255, 0.5) !important;
        transform: translateY(-1px);
    }
    
    /* Code block styling */
    pre, code {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
"""

st.markdown(AEROSPIKE_CSS, unsafe_allow_html=True)


# Helper cached resource loaders
@st.cache_resource
def load_models_and_pipelines():
    """Load trained real-data artifacts; never silently fall back to synthetic data."""
    model_dir = os.path.join(ROOT_DIR, "saved_models")
    required = [
        os.path.join(model_dir, "best_classifier.joblib"),
        os.path.join(model_dir, "feature_extractor.joblib"),
        os.path.join(model_dir, "clustering_anomaly.joblib"),
        os.path.join(model_dir, "model_pipeline.joblib"),
    ]
    if not all(os.path.exists(path) for path in required):
        st.error("⚠️ ScamShield models are not trained yet.")
        st.code(
            "python scripts/setup_real_data.py\n"
            "python scripts/check_dataset.py\n"
            "python scripts/train_models.py --data data/job_offers.csv",
            language="powershell",
        )
        st.info("The final project intentionally does not auto-train on the old synthetic dataset.")
        st.stop()

    model_pipeline = load_pipeline(model_dir=model_dir)
    feat_extractor = FeatureExtractor(max_tfidf_features=300, use_sentence_transformers=False, model_dir=model_dir)
    if not feat_extractor.load_model():
        st.error("Feature extractor artifact could not be loaded. Retrain the model.")
        st.stop()

    classifier = MultiClassFraudClassifier(model_dir=model_dir)
    if not classifier.load_model():
        st.error("Classifier artifact could not be loaded. Retrain the model.")
        st.stop()

    clustering_engine = ClusterAndAnomalyEngine(model_dir=model_dir)
    if not clustering_engine.load_models():
        st.error("Clustering/anomaly artifact could not be loaded. Retrain the model.")
        st.stop()

    rule_miner = AssociationRuleMiner()
    return model_pipeline, feat_extractor, classifier, clustering_engine, rule_miner


model_pipeline, feat_extractor, classifier, clustering_engine, rule_miner = load_models_and_pipelines()


def predict_with_structured_model(feature_matrix):
    """Use the trained combined-feature classifier for website predictions.

    This classifier consumes TF-IDF features plus stipend, experience, contact,
    urgency, financial and domain-related features. The original UI used the
    text-only pipeline for final predictions, causing the website and the
    reported model architecture to disagree.
    """
    predictions = classifier.predict(feature_matrix).astype(object)
    probabilities = classifier.predict_proba(feature_matrix)
    classes = ["Genuine", "Suspicious", "Fraudulent"]
    probability_dicts = []
    for idx, row in enumerate(probabilities):
        p_dict = {cls: float(row[pos]) for pos, cls in enumerate(classes)}
        ordered = sorted(p_dict.items(), key=lambda item: item[1], reverse=True)
        max_conf = ordered[0][1]
        margin = ordered[0][1] - ordered[1][1]
        # Suspicious is an operational uncertainty state. It does not require a
        # separate ground-truth training class when the underlying real dataset
        # is binary (Genuine vs Fraudulent).
        if max_conf < 0.60 or margin < 0.10:
            predictions[idx] = "Suspicious"
        probability_dicts.append(p_dict)
    return predictions, probability_dicts

# Aerospike Top Performance Header
st.markdown("""
<div class="aerospike-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h2 style="margin: 0; font-size: 24px; font-weight: 800; color: #ffffff;">⚡ SCAMSHIELD AI <span style="color: #0066FF; font-weight: 300;">|</span></h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">High-Throughput Student Job & Internship Fraud Verification System</p>
        </div>
        <div style="margin-top: 8px;">
            <span class="aerospike-pill">⚡ ENGINE: IN-MEMORY SCORING</span>
            <span class="aerospike-pill">⏱️ LOCAL ML SCORING</span>
            <span class="aerospike-pill-green">🟢 STATUS: LIVE & READY</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("## 🛡️ ScamShield AI")
st.sidebar.caption("Aerospike Dark Enterprise Edition v2.5")

nav_mode = st.sidebar.radio(
    "Select System Portal:",
    [
        "🎓 Student Verification Portal",
        "📁 Batch Offer Scanner & API Simulator",
        "📊 Placement Officer OLAP Dashboard",
        "🧠 AI Model Performance Center",
        "⚙️ Blacklist & Admin Manager"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
- **NLP Engine**: spaCy / NLTK + TF-IDF
- **Classifier**: Best of Random Forest / Decision Tree / Naive Bayes
- **Outlier Engine**: Isolation Forest
- **Data Mining**: Apriori / FP-Growth
- **OLAP Core**: Multi-Dimensional Cubes
""")


# ==========================================
# PORTAL 1: STUDENT VERIFICATION PORTAL
# ==========================================
if nav_mode == "🎓 Student Verification Portal":
    st.title("🎓 Internship & Job Offer Verification Portal")
    st.write("Submit job offer details, recruitment emails, or WhatsApp messages to compute a **Multi-Factor Trust Score (0-100%)**.")

    # Interactive Step-by-Step Example Walkthrough (Accordion)
    with st.expander("📖 Interactive Guide & Step-by-Step Example (How to Use)", expanded=False):
        st.markdown("""
        ### 🚀 Step-by-Step Example Guide
        
        Follow these steps to analyze any job offer or recruitment communication:
        
        #### 1. Choose a Quick Test Preset or Enter Details
        - **Option A (Presets)**: Click one of the green/yellow/red buttons below to auto-fill realistic test cases.
        - **Option B (Custom Input)**: Paste the job title, recruiter email, company domain, stipend, and offer text into the verification form.
        
        #### 2. Click `🔍 Run ScamShield Verification`
        - The multi-factor trust engine runs TF-IDF text analysis, Random Forest classification, Isolation Forest salary anomaly checks, and active blacklist matching.
        
        #### 3. Interpret Results:
        - 🟢 **80% - 100% (LOW RISK)**: Safe offer. Legitimate corporate recruiter with verified domain.
        - 🟡 **50% - 79% (MODERATE RISK)**: Proceed with caution. Unverified email domain or suspicious urgency.
        - 🔴 **0% - 49% (HIGH RISK / FRAUDULENT)**: **DO NOT APPLY!** Upfront payment, laptop fee, or Telegram scam detected.
        
        ---
        #### 💡 Real-World Example Comparison:
        | Scenario | Recruiter Email | Stipend | Key Text Trigger | Expected Score & Result |
        | :--- | :--- | :--- | :--- | :--- |
        | **Genuine Offer** | `careers@techcorp.com` | ₹25,000 | Official coding test & interview process | 🟢 **95% (LOW RISK)** |
        | **Registration Fee Scam** | `scam-recruiter@gmail.com` | ₹65,000 | "Pay ₹2,500 registration fee to UPI" | 🔴 **12% (HIGH RISK)** |
        | **Telegram Phishing Scam** | `wa_recruiter@gmail.com` | ₹45,000 | "DM on Telegram group, refundable deposit" | 🔴 **28% (HIGH RISK)** |
        """)

    st.markdown("### ⚡ Quick Test Presets")
    col_p1, col_p2, col_p3 = st.columns(3)

    sample_preset = None
    if col_p1.button("🟢 Load Genuine Offer Sample"):
        sample_preset = "GENUINE"
    if col_p2.button("🟡 Load Registration Fee Scam"):
        sample_preset = "FEE_SCAM"
    if col_p3.button("🔴 Load Telegram Phishing Scam"):
        sample_preset = "TELEGRAM_SCAM"

    # Default Form Values based on sample
    default_title = "Cloud Operations Software Intern"
    default_company = "TechCorp Systems India"
    default_domain = "techcorp.com"
    default_email = "careers@techcorp.com"
    default_phone = "+91-9847123456"
    default_dept = "CS/IT"
    default_stipend = 25000.0
    default_exp = 0.0
    default_text = "We are hiring a Software Engineering Intern for our cloud infrastructure team. Responsibilities include building Python REST APIs and working with AWS cloud services. Requirements: B.Tech CS/IT student, basic coding proficiency. Standard official interview process."

    if sample_preset == "FEE_SCAM":
        default_title = "Data Entry & Web Researcher"
        default_company = "EarnFromHome Ltd"
        default_domain = "pay-registration-jobs.biz"
        default_email = "scam-recruiter@gmail.com"
        default_phone = "+91-9876543210"
        default_dept = "CS/IT"
        default_stipend = 65000.0
        default_exp = 0.0
        default_text = "CONGRATULATION! DIRECT SELECTION OFFER LETTER ISSUED! Earn Rs 65,000 per month typing captchas. NO INTERVIEW REQUIRED! To receive employee ID and laptop, pay mandatory registration fee of Rs 2,500 to UPI fastpay@upi today itself! Apply on WhatsApp."
    elif sample_preset == "TELEGRAM_SCAM":
        default_title = "Part Time Social Media Manager"
        default_company = "Quick Hire Network"
        default_domain = "quickhire-jobs.wixsite.com"
        default_email = "wa_recruiter_2026@gmail.com"
        default_phone = "+91-9123456789"
        default_dept = "Business & Finance"
        default_stipend = 45000.0
        default_exp = 0.0
        default_text = "Urgent hiring! Earn daily income working 1 hour from phone. No interview needed. Contact admin on Telegram group. Refundable security deposit of Rs 1,500 required for software license."

    with st.form("scan_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.text_input("Job / Internship Title:", value=default_title)
            company_name = st.text_input("Company Name:", value=default_company)
            department = st.selectbox("Department:", ["CS/IT", "Electronics & Comm", "Mechanical", "Civil", "Biotechnology", "Business & Finance"], index=0 if default_dept=="CS/IT" else 5)
        with col2:
            company_domain = st.text_input("Company Website / Domain:", value=default_domain)
            contact_email = st.text_input("Recruiter Email:", value=default_email)
            contact_phone = st.text_input("Contact Phone Number:", value=default_phone)
        with col3:
            stipend = st.number_input("Monthly Stipend / Salary (INR):", min_value=0.0, max_value=500000.0, value=default_stipend, step=5000.0)
            exp_years = st.number_input("Required Experience (Years):", min_value=0.0, max_value=15.0, value=default_exp, step=0.5)
            location = st.selectbox("Job Location:", ["Remote", "Kochi", "Bangalore", "Trivandrum", "Chennai", "Hybrid"])

        upload = st.file_uploader("Offer Letter PDF/Image:", type=["pdf", "png", "jpg", "jpeg"])
        extracted_upload_text = ""
        if upload is not None:
            try:
                extracted_upload_text = extract_text_from_upload(upload, suffix=os.path.splitext(upload.name)[1])
                st.success("Offer letter text extracted.")
            except Exception as exc:
                st.warning(f"Could not extract upload text: {exc}")

        job_description = st.text_area("Job Offer Description / Email Body Text:", value=default_text + ("\n" + extracted_upload_text if extracted_upload_text else ""), height=130)

        submit_btn = st.form_submit_button("🔍 Run ScamShield Verification")

    if submit_btn or sample_preset is not None:
        db = SessionLocal()

        # 1. Fetch active blacklists
        blacklists = db.query(BlacklistEntity).all()
        bl_dicts = [{"type": b.entity_type, "value": b.entity_value, "reason": b.reason} for b in blacklists]

        matched_bl = find_blacklist_matches(
            text=job_description,
            domain=company_domain,
            email=contact_email,
            company_name=company_name,
            phone=contact_phone,
            blacklist_entries=bl_dicts,
        )

        # 2. Extract Features & Predict
        sample_df = pd.DataFrame([{
            'description': job_description,
            'stipend': stipend,
            'exp_years': exp_years,
            'company_domain': company_domain,
            'contact_email': contact_email
        }])

        X_vec, _ = feat_extractor.prepare_feature_matrix(sample_df, fit=False)
        num_vec = X_vec[:, -12:]

        preds_main, probs_main = predict_with_structured_model(X_vec)
        pred_class = str(preds_main[0])
        prob_dict = probs_main[0]

        # 3. Cluster & Anomaly
        cluster_id = int(clustering_engine.predict_cluster(X_vec)[0])
        cluster_name = clustering_engine.get_cluster_name(cluster_id)

        is_anom, anom_scores = clustering_engine.detect_anomaly(num_vec)
        is_anomaly = bool(is_anom[0])
        anom_score = float(anom_scores[0])

        # 4. Urgency Analysis & Trust Score
        urgency_res = analyze_urgency_and_suspicion(job_description, domain=company_domain, email=contact_email)
        trust_engine = TrustScoreEngine()

        res = trust_engine.compute_trust_score(
            predicted_class=pred_class,
            class_probabilities=prob_dict,
            urgency_analysis=urgency_res,
            is_anomaly=is_anomaly,
            anomaly_score=anom_score,
            blacklist_matches=matched_bl,
            raw_text=job_description
        )

        score = res["trust_score"]
        status_color = res["status_color"]
        risk_level = res["risk_level"]

        # Save scan result to Database
        offer_record = ScannedOffer(
            title=title,
            company_name=company_name,
            company_domain=company_domain,
            contact_email=contact_email,
            contact_phone=contact_phone,
            department=department,
            location=location,
            stipend=stipend,
            exp_years=exp_years,
            description=job_description,
            predicted_class=pred_class,
            trust_score=score,
            archetype_cluster=cluster_id,
            is_anomaly=is_anomaly,
            anomaly_score=anom_score,
            urgency_index=urgency_res["urgency_index"],
            risk_factors_json=json.dumps(res["explanation_alerts"])
        )
        db.add(offer_record)
        db.add(AnalyticsLog(event_type="STUDENT_SCAN", details=f"Scanned '{title}' at {company_name}. Score: {score}% ({pred_class})"))
        db.commit()
        db.close()

        st.markdown("### Feedback")
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        feedback_payload = {
            "title": title,
            "company_name": company_name,
            "company_domain": company_domain,
            "contact_email": contact_email,
            "contact_phone": contact_phone,
            "description": job_description,
            "predicted_class": pred_class,
            "trust_score": score,
        }
        if fb_col1.button("Result Correct"):
            append_review_item({**feedback_payload, "feedback": "correct"}, REVIEW_QUEUE_PATH)
            st.success("Feedback recorded.")
        if fb_col2.button("Result Incorrect"):
            append_review_item({**feedback_payload, "feedback": "incorrect"}, REVIEW_QUEUE_PATH)
            st.warning("Added to review queue.")
        if fb_col3.button("Report as Scam"):
            append_review_item({**feedback_payload, "feedback": "reported_scam"}, REVIEW_QUEUE_PATH)
            st.error("Scam report added to review queue.")

        # Display Results
        st.divider()
        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            st.markdown("### 🎯 Multi-Factor Trust Score Gauge")
            
            gauge_color = "#16a34a" if score >= 80 else ("#d97706" if score >= 50 else "#dc2626")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"<b>{risk_level}</b>", 'font': {'size': 22, 'color': gauge_color}},
                number={'suffix': "%", 'font': {'size': 44, 'color': gauge_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#111827",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(220, 38, 38, 0.25)'},
                        {'range': [50, 80], 'color': 'rgba(217, 119, 6, 0.25)'},
                        {'range': [80, 100], 'color': 'rgba(22, 163, 74, 0.25)'}
                    ],
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ffffff"})
            st.plotly_chart(fig, use_container_width=True)

            if score >= 80:
                st.success(f"💡 **Recommendation**: {res['recommendation']}")
            elif score >= 50:
                st.warning(f"💡 **Recommendation**: {res['recommendation']}")
            else:
                st.error(f"💡 **Recommendation**: {res['recommendation']}")

        with res_col2:
            st.markdown("### 📊 Multi-Factor Analysis Breakdown")
            
            m1, m2 = st.columns(2)
            with m1:
                badge_class = "metric-badge-green" if pred_class == "Genuine" else ("metric-badge-yellow" if pred_class == "Suspicious" else "metric-badge-red")
                st.markdown(f"**ML Classification:** <span class='{badge_class}'>{pred_class}</span>", unsafe_allow_html=True)
                st.caption(f"Model Confidence: {round(prob_dict[pred_class]*100, 1)}%")
            with m2:
                st.markdown(f"**Archetype Cluster:** `{cluster_name}`")
                st.caption(f"Cluster ID: #{cluster_id}")

            st.markdown("---")
            m3, m4 = st.columns(2)
            with m3:
                bl_status = f"🚨 MATCH ({len(matched_bl)})" if matched_bl else "✅ CLEAN"
                st.markdown(f"**Blacklist Status:** `{bl_status}`")
            with m4:
                anom_status = f"⚠️ ANOMALOUS ({round(anom_score*100, 1)}%)" if is_anomaly else "✅ NORMAL"
                st.markdown(f"**Salary/Exp Outlier:** `{anom_status}`")

            st.markdown("#### 🤖 ML Probability Distribution")
            prob_df = pd.DataFrame({
                "Class": list(prob_dict.keys()),
                "Probability (%)": [round(v * 100, 1) for v in prob_dict.values()],
            })
            fig_prob = px.bar(
                prob_df,
                x="Class",
                y="Probability (%)",
                range_y=[0, 100],
                text="Probability (%)",
                title="Model Confidence by Risk Class",
            )
            fig_prob.update_traces(textposition="outside")
            fig_prob.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=45, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ffffff"},
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Risk Explanations Card
        st.markdown("### 💬 Risk Alert Cards & Explanations")
        if res["explanation_alerts"]:
            for alert in res["explanation_alerts"]:
                if "CRITICAL" in alert or "Upfront payment" in alert or "High risk classification" in alert:
                    st.error(f"🚨 **{alert}**")
                else:
                    st.warning(f"⚠️ **{alert}**")
        
        if res["bonuses"]:
            for bonus in res["bonuses"]:
                st.success(f"✅ **{bonus}**")


# ==========================================
# PORTAL 2: BATCH SCANNER & API SIMULATOR
# ==========================================
elif nav_mode == "📁 Batch Offer Scanner & API Simulator":
    st.title("📁 Batch Offer Scanner & REST API Payload Simulator")
    st.write("Scan multiple job offers in bulk via CSV upload, or simulate real-time JSON payloads for integration.")

    tab_b1, tab_b2 = st.tabs(["📁 Bulk CSV Batch Scanner", "⚡ REST API Payload Simulator"])

    with tab_b1:
        st.subheader("📁 Bulk CSV Job Offer Scanner")
        st.write("Upload a CSV containing columns: `title`, `company_name`, `company_domain`, `contact_email`, `stipend`, `exp_years`, `description`.")

        sample_csv_data = pd.DataFrame([
            {
                "title": "Backend Python Intern",
                "company_name": "Acme Technologies",
                "company_domain": "acme.com",
                "contact_email": "careers@acme.com",
                "stipend": 20000,
                "exp_years": 0,
                "description": "Building Python microservices and PostgreSQL databases. Official tech assessment required."
            },
            {
                "title": "Captchas Typing Assistant",
                "company_name": "Easy Money Corp",
                "company_domain": "pay-fee-jobs.site",
                "contact_email": "scamhr@gmail.com",
                "stipend": 75000,
                "exp_years": 0,
                "description": "Pay Rs 2000 registration fee to UPI fastpay@upi today itself. No interview needed."
            }
        ])

        col_csv1, col_csv2 = st.columns([1, 1])
        with col_csv1:
            uploaded_file = st.file_uploader("Choose CSV File:", type=["csv"])
        with col_csv2:
            st.markdown("**Need a sample template?**")
            st.download_button(
                "📥 Download Sample CSV Template",
                data=sample_csv_data.to_csv(index=False),
                file_name="sample_job_offers_batch.csv",
                mime="text/csv"
            )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded CSV with {len(batch_df)} offer records.")
                
                if st.button("🚀 Process Batch Verification"):
                    results_list = []
                    trust_engine = TrustScoreEngine()

                    db = SessionLocal()
                    bl_dicts = [{"type": b.entity_type, "value": b.entity_value, "reason": b.reason} for b in db.query(BlacklistEntity).all()]
                    db.close()

                    X_batch, _ = feat_extractor.prepare_feature_matrix(batch_df, fit=False)
                    num_batch = X_batch[:, -12:]
                    texts = batch_df["description"].fillna("").astype(str).tolist()
                    preds, probs_list = predict_with_structured_model(X_batch)
                    clusters = clustering_engine.predict_cluster(X_batch)
                    anoms, anom_scores = clustering_engine.detect_anomaly(num_batch)

                    for idx, row in batch_df.iterrows():
                        pred_c = str(preds[idx])
                        p_dict = probs_list[idx]
                        
                        desc = str(row.get('description', ''))
                        dom = str(row.get('company_domain', ''))
                        em = str(row.get('contact_email', ''))

                        matched_bl = find_blacklist_matches(
                            text=desc,
                            domain=dom,
                            email=em,
                            company_name=str(row.get("company_name", "")),
                            phone=str(row.get("contact_phone", "")),
                            blacklist_entries=bl_dicts,
                        )
                        urg_res = analyze_urgency_and_suspicion(desc, domain=dom, email=em)

                        sc_res = trust_engine.compute_trust_score(
                            predicted_class=pred_c,
                            class_probabilities=p_dict,
                            urgency_analysis=urg_res,
                            is_anomaly=bool(anoms[idx]),
                            anomaly_score=float(anom_scores[idx]),
                            blacklist_matches=matched_bl,
                            raw_text=desc
                        )

                        results_list.append({
                            "Title": row.get('title', 'N/A'),
                            "Company": row.get('company_name', 'N/A'),
                            "Predicted Class": pred_c,
                            "Trust Score (%)": sc_res['trust_score'],
                            "Risk Level": sc_res['risk_level'],
                            "Archetype Cluster": clustering_engine.get_cluster_name(int(clusters[idx])),
                            "Blacklist Matches": len(matched_bl),
                            "Recommendation": sc_res['recommendation']
                        })

                    res_df = pd.DataFrame(results_list)
                    st.subheader("📊 Batch Scan Results Summary")
                    st.dataframe(res_df, use_container_width=True)

                    st.download_button(
                        "📥 Download Batch Verification Report (CSV)",
                        data=res_df.to_csv(index=False),
                        file_name="scamshield_batch_report.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

    with tab_b2:
        st.subheader("⚡ REST API Payload Simulator")
        st.write("Simulate real-time POST payload request to `https://api.scamshield.ai/v1/verify`.")

        sample_json_input = {
            "title": "Part Time Content Writer",
            "company_name": "Quick Copy Solutions",
            "company_domain": "quickcopy-careers.info",
            "contact_email": "scamhr@gmail.com",
            "contact_phone": "+91-9876543210",
            "stipend": 55000.0,
            "exp_years": 0.0,
            "description": "Urgent hiring! Earn 55000 stipend typing articles. No interview. Pay registration fee of Rs 2500 to fastpay@upi."
        }

        api_input_str = st.text_area("JSON API Request Body:", value=json.dumps(sample_json_input, indent=2), height=220)

        if st.button("⚡ Execute API Request"):
            try:
                payload = json.loads(api_input_str)
                df_api = pd.DataFrame([payload])

                X_api, _ = feat_extractor.prepare_feature_matrix(df_api, fit=False)
                num_api = X_api[:, -12:]
                desc = payload.get('description', '')
                preds_api, probs_api = predict_with_structured_model(X_api)
                pred_c = str(preds_api[0])
                p_dict = probs_api[0]

                cl_id = int(clustering_engine.predict_cluster(X_api)[0])
                anom_flag, anom_scores = clustering_engine.detect_anomaly(num_api)

                db = SessionLocal()
                bl_dicts = [{"type": b.entity_type, "value": b.entity_value, "reason": b.reason} for b in db.query(BlacklistEntity).all()]
                db.close()

                dom = payload.get('company_domain', '')
                em = payload.get('contact_email', '')
                matched_bl = find_blacklist_matches(
                    text=desc,
                    domain=dom,
                    email=em,
                    company_name=str(payload.get("company_name", "")),
                    phone=str(payload.get("contact_phone", "")),
                    blacklist_entries=bl_dicts,
                )
                urg_res = analyze_urgency_and_suspicion(desc, domain=dom, email=em)

                trust_engine = TrustScoreEngine()
                sc_res = trust_engine.compute_trust_score(
                    predicted_class=pred_c,
                    class_probabilities=p_dict,
                    urgency_analysis=urg_res,
                    is_anomaly=bool(anom_flag[0]),
                    anomaly_score=float(anom_scores[0]),
                    blacklist_matches=matched_bl,
                    raw_text=desc
                )

                api_response = {
                    "status": "200_OK",
                    "latency_ms": 1.15,
                    "scamshield_version": "2.5.0",
                    "verification": {
                        "trust_score": sc_res['trust_score'],
                        "risk_level": sc_res['risk_level'],
                        "predicted_class": pred_c,
                        "class_probabilities": p_dict,
                        "recommendation": sc_res['recommendation'],
                        "archetype_cluster": clustering_engine.get_cluster_name(cl_id),
                        "is_outlier_anomaly": bool(anom_flag[0]),
                        "blacklist_matches": matched_bl,
                        "explanation_alerts": sc_res['explanation_alerts']
                    }
                }

                st.markdown("### 📥 API Response JSON:")
                st.code(json.dumps(api_response, indent=2), language="json")

            except Exception as ex:
                st.error(f"Invalid JSON payload: {str(ex)}")


# ==========================================
# PORTAL 3: PLACEMENT OFFICER OLAP DASHBOARD
# ==========================================
elif nav_mode == "📊 Placement Officer OLAP Dashboard":
    st.title("📊 Placement Officer OLAP Analytics Dashboard")
    st.write("Perform **Data Mining OLAP Operations** (Roll-up, Drill-down, Slice & Dice) on historical offer scan logs.")

    olap = OLAPAnalyticsEngine()
    df_offers = olap.load_offers_dataframe()

    if df_offers.empty:
        st.warning("No scanned offers recorded yet. Running training pipeline...")
        from scripts.train_models import run_training_pipeline
        run_training_pipeline()
        df_offers = olap.load_offers_dataframe()

    # Ensure defensive fields exist
    df_offers['description'] = df_offers.get('description', pd.Series(['']*len(df_offers))).fillna('')
    df_offers['company_domain'] = df_offers.get('company_domain', pd.Series(['']*len(df_offers))).fillna('')
    df_offers['contact_email'] = df_offers.get('contact_email', pd.Series(['']*len(df_offers))).fillna('')

    # KPI Summary Cards
    total_scans = len(df_offers)
    fraud_count = len(df_offers[df_offers['predicted_class'] == 'Fraudulent'])
    susp_count = len(df_offers[df_offers['predicted_class'] == 'Suspicious'])
    gen_count = len(df_offers[df_offers['predicted_class'] == 'Genuine'])
    fraud_pct = round((fraud_count / max(1, total_scans)) * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Offers Scanned", total_scans)
    k2.metric("Genuine Offers", f"{gen_count} ({round(gen_count/max(1,total_scans)*100,1)}%)")
    k3.metric("Suspicious Offers", f"{susp_count} ({round(susp_count/max(1,total_scans)*100,1)}%)")
    k4.metric("Fraudulent Offers", f"{fraud_count} ({fraud_pct}%)")

    st.divider()

    # OLAP Tabbed Interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time-Series Scam Trends (Roll-Up)",
        "🌌 Archetype Clusters & PCA Map (2D/3D)",
        "⛓️ Mined Association Rules",
        "🧩 Departmental Slice & Dice"
    ])

    # TAB 1: ROLL-UP TIME SERIES
    with tab1:
        st.subheader("📈 Scam Trends over Time (OLAP Roll-Up)")
        freq_choice = st.selectbox("Roll-Up Frequency Level:", ["Daily ('D')", "Weekly ('W')", "Monthly ('M')"], index=1)
        freq_code = 'D' if 'Daily' in freq_choice else ('W' if 'Weekly' in freq_choice else 'ME')

        rollup_df = olap.rollup_time_series(freq=freq_code)
        if not rollup_df.empty:
            fig_trend = px.bar(
                rollup_df,
                x='posting_date',
                y=[c for c in ['Genuine', 'Suspicious', 'Fraudulent'] if c in rollup_df.columns],
                title="Scam Volume Roll-Up Trend over Time",
                labels={'posting_date': 'Date Period', 'value': 'Offer Count'},
                color_discrete_map={'Genuine': '#16a34a', 'Suspicious': '#d97706', 'Fraudulent': '#dc2626'}
            )
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='#ffffff'))
            st.plotly_chart(fig_trend, use_container_width=True)

    # TAB 2: CLUSTERING & PCA
    with tab2:
        st.subheader("🌌 K-Means Archetype Clusters & PCA Projection")
        st.write("Visualizing job offers projected onto Principal Component Analysis (PCA) axes.")

        pca_mode = st.radio("PCA Visualization Dimension:", ["2D Scatter Plot", "3D Scatter Plot"], horizontal=True)

        try:
            X_mat, _ = feat_extractor.prepare_feature_matrix(df_offers, fit=False)
            coords_2d, coords_3d = clustering_engine.transform_pca(X_mat)

            df_pca = df_offers.copy()
            df_pca['Cluster_Name'] = df_pca['archetype_cluster'].apply(lambda value: clustering_engine.get_cluster_name(int(value)))

            if pca_mode == "2D Scatter Plot":
                df_pca['PCA_1'] = coords_2d[:, 0]
                df_pca['PCA_2'] = coords_2d[:, 1]
                fig_pca = px.scatter(
                    df_pca,
                    x='PCA_1',
                    y='PCA_2',
                    color='Cluster_Name',
                    symbol='predicted_class',
                    size='stipend',
                    hover_data=['title', 'company_name', 'trust_score', 'department'],
                    title="2D Job Offer Archetype Clusters & Outliers",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='#ffffff'), height=520)
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                df_pca['PCA_1'] = coords_3d[:, 0]
                df_pca['PCA_2'] = coords_3d[:, 1]
                df_pca['PCA_3'] = coords_3d[:, 2]
                fig_pca3 = px.scatter_3d(
                    df_pca,
                    x='PCA_1',
                    y='PCA_2',
                    z='PCA_3',
                    color='Cluster_Name',
                    symbol='predicted_class',
                    hover_data=['title', 'company_name', 'trust_score'],
                    title="3D Job Offer Feature Embeddings PCA Scatter",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_pca3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color='#ffffff'), height=580)
                st.plotly_chart(fig_pca3, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating PCA cluster plot: {str(e)}")

    # TAB 3: ASSOCIATION RULES
    with tab3:
        st.subheader("⛓️ Mined Association Rules (Apriori / FP-Growth)")
        st.write("Association rules discovered from job posting patterns leading to `{Fraudulent}` or `{Suspicious}` outcomes.")

        mined_rules_df = rule_miner.mine_rules(df_offers)
        top_fraud = rule_miner.get_top_fraud_rules(top_n=10)

        if top_fraud:
            rules_table = pd.DataFrame(top_fraud)
            st.dataframe(
                rules_table.style.format({'support': '{:.3f}', 'confidence': '{:.3f}', 'lift': '{:.2f}'}),
                use_container_width=True
            )
        else:
            st.info("Mining association rules...")

    # TAB 4: DEPARTMENT SLICE & DICE
    with tab4:
        st.subheader("🧩 Departmental Vulnerability Slice & Dice")

        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            sel_depts = st.multiselect("Select Departments (Slice & Dice):", df_offers['department'].unique().tolist(), default=df_offers['department'].unique().tolist()[:3])
        with c_filter2:
            sel_classes = st.multiselect("Select Risk Classes:", ["Genuine", "Suspicious", "Fraudulent"], default=["Suspicious", "Fraudulent"])

        subcube = olap.dice_subcube(departments=sel_depts, fraud_types=sel_classes)

        st.markdown(f"**Filtered Sub-Cube Records ({len(subcube)} offers):**")
        st.dataframe(
            subcube[['title', 'company_name', 'department', 'stipend', 'predicted_class', 'trust_score', 'posting_date']],
            use_container_width=True
        )

        st.subheader("📊 Department vs. Fraud Class Pivot Table")
        pivot_df = olap.pivot_dept_vs_fraud()
        st.dataframe(pivot_df, use_container_width=True)


# ==========================================
# PORTAL 4: AI MODEL PERFORMANCE CENTER
# ==========================================
elif nav_mode == "🧠 AI Model Performance Center":
    st.title("🧠 AI Model Evaluation & Performance Center")
    st.write("Detailed performance metrics, confusion matrices, and explainable feature importances for trained machine learning models.")

    eval_results = getattr(classifier, "evaluation_results", {})

    if not eval_results:
        st.info("Loading model metrics...")

    col_m1, col_m2 = st.columns([1.2, 1])

    with col_m1:
        st.subheader("🏆 Fraud Model Comparison (test set untouched during training)")
        m_data = []
        for name, res in eval_results.items():
            m_data.append({
                "Model Algorithm": name,
                "Accuracy (%)": f"{res['accuracy']*100:.2f}%",
                "Fraud Precision": f"{res.get('precision_fraud', 0)*100:.1f}%",
                "Fraud Recall": f"{res.get('recall_fraud', 0)*100:.1f}%",
                "Fraud F1": f"{res.get('f1_fraud', 0):.3f}",
                "PR-AUC": f"{res.get('pr_auc', 0):.3f}",
                "Threshold": f"{res.get('threshold', 0.50):.2f}",
                "Selected Best": "✅ BEST MODEL" if name == classifier.best_model_name else "Candidate"
            })
        st.dataframe(pd.DataFrame(m_data), use_container_width=True)

        # Feature Importance Chart
        st.subheader("🔍 Top 15 Explainable Features")
        db_df = OLAPAnalyticsEngine().load_offers_dataframe()
        if not db_df.empty:
            _, feat_names = feat_extractor.prepare_feature_matrix(db_df, fit=False)
            top_feats = classifier.get_feature_importances(feat_names)
            if top_feats:
                df_top_f = pd.DataFrame(top_feats[:15])
                fig_imp = px.bar(
                    df_top_f,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top Feature Importance Rankings",
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='#ffffff'), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)

    with col_m2:
        st.subheader("📊 Archetype Cluster Breakdown")
        db_df = OLAPAnalyticsEngine().load_offers_dataframe()
        if not db_df.empty:
            cluster_counts = db_df['archetype_cluster'].apply(lambda value: clustering_engine.get_cluster_name(int(value))).value_counts().reset_index()
            cluster_counts.columns = ['Archetype', 'Count']
            fig_pie = px.pie(
                cluster_counts,
                names='Archetype',
                values='Count',
                title="Job Offer Archetype Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color='#ffffff'))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("🎯 Final Grouped Holdout Metrics")
    best_metrics = eval_results.get(classifier.best_model_name, {})
    gm1, gm2, gm3, gm4 = st.columns(4)
    gm1.metric("Fraud Precision", f"{best_metrics.get('precision_fraud', 0.0)*100:.1f}%")
    gm2.metric("Fraud Recall", f"{best_metrics.get('recall_fraud', 0.0)*100:.1f}%")
    gm3.metric("Fraud F1", f"{best_metrics.get('f1_fraud', 0.0):.3f}")
    gm4.metric("PR-AUC", f"{best_metrics.get('pr_auc', 0.0):.3f}")
    cm = best_metrics.get("confusion_matrix")
    if cm:
        cm_fig = px.imshow(
            cm,
            x=["Genuine", "Fraudulent"],
            y=["Genuine", "Fraudulent"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Grouped Held-Out Test Confusion Matrix",
        )
        cm_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='#ffffff'))
        st.plotly_chart(cm_fig, use_container_width=True)


# ==========================================
# PORTAL 5: BLACKLIST & ADMIN MANAGER
# ==========================================
elif nav_mode == "⚙️ Blacklist & Admin Manager":
    st.title("⚙️ ScamShield Blacklist & Admin Management Portal")
    st.write("Manage blacklisted domains, recruiters, companies, phone numbers, and payment handles. Blacklisted entities immediately trigger critical penalties.")

    db = SessionLocal()

    col_bl1, col_bl2 = st.columns([1.2, 1])

    with col_bl1:
        st.subheader("📋 Active Blacklist Records")
        b_records = db.query(BlacklistEntity).all()
        if b_records:
            b_data = [{"ID": b.id, "Type": b.entity_type, "Value": b.entity_value, "Reason": b.reason, "Added At": b.added_at.strftime('%Y-%m-%d')} for b in b_records]
            st.dataframe(pd.DataFrame(b_data), use_container_width=True)
            delete_id = st.number_input("Blacklist ID to delete:", min_value=0, value=0, step=1)
            if st.button("Delete Blacklist Entry") and delete_id:
                target = db.query(BlacklistEntity).filter(BlacklistEntity.id == int(delete_id)).first()
                if target:
                    db.delete(target)
                    db.add(AnalyticsLog(event_type="BLACKLIST_DELETE", details=f"Deleted blacklist ID {delete_id}"))
                    db.commit()
                    st.success("Blacklist entry deleted.")
                    st.rerun()
                else:
                    st.warning("No blacklist entry found for that ID.")
        else:
            st.info("No blacklisted entities present.")

    with col_bl2:
        st.subheader("➕ Add New Blacklist Entity")
        with st.form("add_bl_form"):
            e_type = st.selectbox("Entity Type:", ["DOMAIN", "COMPANY", "EMAIL", "PHONE", "PAYMENT_HANDLE"])
            e_val = st.text_input("Entity Value (e.g. scamdomain.com or fakehr@gmail.com):")
            e_reason = st.text_input("Reason / Student Report Details:", value="Reported by students for upfront registration fee scam.")

            add_btn = st.form_submit_button("🛡️ Blacklist Entity")

            if add_btn and e_val:
                try:
                    new_bl = BlacklistEntity(entity_type=e_type, entity_value=e_val.strip(), reason=e_reason.strip())
                    db.add(new_bl)
                    db.add(AnalyticsLog(event_type="BLACKLIST_ADD", details=f"Added {e_type}: {e_val}"))
                    db.commit()
                    st.success(f"✅ Successfully blacklisted {e_val}!")
                    st.rerun()
                except Exception as ex:
                    db.rollback()
                    st.error(f"Error adding blacklist entity: {str(ex)}")

    st.divider()
    st.subheader("Review Queue")
    review_items = load_review_queue(REVIEW_QUEUE_PATH)
    if review_items:
        st.dataframe(pd.DataFrame(review_items), use_container_width=True)
    else:
        st.info("No feedback reports are waiting for review.")

    db.close()
