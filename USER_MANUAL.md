# 🛡️ ScamShield AI (SkillMine) — Final User Manual

## 1. System Purpose

ScamShield AI is a web-based job and internship offer verification system. It combines:

- TF-IDF text features
- Structured job-offer features such as stipend and experience
- ML classification
- Rule-based scam indicators
- Blacklist matching
- Salary/experience anomaly detection
- Explainable risk alerts
- Batch CSV scanning
- OLAP analytics
- Association-rule mining

**Important:** The Trust Score is a risk estimate. It is not a legal guarantee that an offer is safe or fraudulent.

---

## 2. Start the Application

Open a terminal inside the `SkillMine` project folder.

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Check the training data

```bash
python scripts/check_dataset.py
```

A healthy dataset should contain many genuinely different job descriptions. A very high duplicate percentage means the model can memorize templates instead of learning general fraud patterns.

### Train the models

For the included demonstration dataset:

```bash
python scripts/seed_data.py
python scripts/train_models.py
```

For a custom CSV:

```bash
python scripts/train_models.py --data path\to\your_dataset.csv
```

The trainer prefers `data/job_offers.csv` when it exists.

### Start the website

```bash
streamlit run dashboard/streamlit_app.py
```

Open the local Streamlit address shown in the terminal.

---

## 3. Student Verification Portal

Use this portal to verify one job or internship offer at a time.

### Step 1 — Enter the offer details

Fill in:

- Job / Internship Title
- Company Name
- Company Website / Domain
- Recruiter Email
- Contact Phone
- Department
- Job Location
- Monthly Stipend / Salary
- Required Experience
- Job Offer Description / Email Body

### Step 2 — Upload an offer letter

Use the upload field for:

- PDF
- PNG
- JPG
- JPEG

The system attempts to extract text and append it to the offer description.

### Step 3 — Run verification

Click:

**🔍 Run ScamShield Verification**

The system then:

1. Extracts text features.
2. Generates structured risk features.
3. Runs the trained ML classifier.
4. Checks blacklist records.
5. Detects payment and urgency indicators.
6. Checks numerical anomalies.
7. Calculates the final Trust Score.
8. Stores the scan in the local database.

---

## 4. Understanding the Result

### Trust Score

The final score is between 0 and 100.

| Score | Risk Level | Meaning |
|---|---|---|
| 75–100 | LOW RISK | Few strong scam indicators were found |
| 50–74.9 | MODERATE RISK | Evidence is mixed; verify independently |
| 0–49.9 | HIGH RISK / FRAUDULENT | Strong fraud evidence is present |

A higher Trust Score means lower estimated risk.

### ML Classification

The model returns one of:

- **Genuine** — model sees stronger legitimate patterns.
- **Suspicious** — evidence is mixed or the prediction is uncertain.
- **Fraudulent** — model sees strong fraud patterns.

### Model Confidence

This shows the probability assigned by the trained classifier. The application should display the actual model probabilities rather than forcing every result to 85%.

### ML Probability Distribution

The bar chart shows the probability assigned to:

- Genuine
- Suspicious
- Fraudulent

Different input text and features should produce different probability distributions.

### Blacklist Status

`✅ CLEAN` means no active blacklist match was found.

`🚨 MATCH` means the submitted domain, email, company, phone number, payment handle, or related text matched an active blacklist record.

### Salary / Experience Outlier

The Isolation Forest checks whether the numerical combination of offer features is unusual compared with the training distribution.

Anomaly detection is supporting evidence; it should not be treated as proof of fraud by itself.

---

## 5. Risk Alert Cards

The alert section explains why the score changed.

Typical alerts include:

- Upfront payment or registration fee
- Suspicious communication channel
- Excessive urgency
- Bypassed interview process
- Blacklisted entity
- Statistical stipend/experience anomaly
- Low model confidence

Read the alerts rather than looking only at the final percentage.

---

## 6. Recommended Safety Behavior

For a low-risk result:

1. Open the employer's official website manually.
2. Verify that the job exists on the official careers page.
3. Verify the recruiter identity.
4. Confirm the interview process.
5. Never share passwords, OTPs, PINs, or banking credentials.

For a moderate-risk result:

- Perform independent verification.
- Contact the company using contact information from its official website.
- Do not transfer money.

For a high-risk result:

- Do not pay registration, training, laptop, security, or processing fees.
- Do not send OTPs, passwords, or banking credentials.
- Stop interacting until the employer is independently verified.

---

## 7. Quick Test Presets

The Student Verification Portal contains demo presets.

### 🟢 Genuine Offer Sample

Loads a professional internship-style offer with:

- Corporate-style email
- Standard hiring process
- Technical assessment/interview
- No upfront fee language

Use it to verify that the system produces a high Trust Score.

### 🟡 Registration Fee Scam

Loads an offer containing:

- Direct selection
- No interview
- High stipend
- Registration fee
- UPI/payment language

Use it to test the fraud detection path.

### 🔴 Telegram Phishing Scam

Loads an offer containing:

- Urgent hiring
- Telegram/WhatsApp communication
- No interview
- Deposit or fee language

Use it to test communication-channel and financial-risk rules.

**Demo scores are not fixed expected values. They depend on the currently trained model and training data.**

---

## 8. Batch Offer Scanner

Open:

**📁 Batch Offer Scanner & API Simulator**

Upload a CSV containing fields such as:

```text
title
company_name
company_domain
contact_email
contact_phone
stipend
exp_years
description
```

Then click:

**🚀 Process Batch Verification**

The results table contains:

- Title
- Company
- Predicted Class
- Trust Score
- Risk Level
- Archetype Cluster
- Blacklist Matches
- Recommendation

Use:

**📥 Download Batch Verification Report (CSV)**

to save the results.

---

## 9. REST API Payload Simulator

This section demonstrates the JSON structure expected by a future REST API.

Paste a JSON object and click:

**⚡ Execute API Request**

The simulator returns:

- Trust Score
- Risk Level
- Predicted Class
- Class Probabilities
- Recommendation
- Archetype Cluster
- Outlier Status
- Blacklist Matches
- Explanation Alerts

This page is a simulator inside the Streamlit website; it is not, by itself, a deployed public REST service.

---

## 10. Placement Officer OLAP Dashboard

Open:

**📊 Placement Officer OLAP Dashboard**

This page is for historical analysis of scanned offers.

### Time-Series Scam Trends

Select:

- Daily
- Weekly
- Monthly

Use this view to study changes in genuine, suspicious, and fraudulent offer volume over time.

### Archetype Clusters & PCA

The PCA view plots offers according to their feature representation.

Use it to identify groups of similar offers and unusual points.

### Mined Association Rules

Association rules show recurring combinations of indicators.

Key values:

- Support
- Confidence
- Lift

Higher lift can indicate a stronger association than random co-occurrence.

### Departmental Slice & Dice

Use the department and class filters to analyze subsets of historical data.

The pivot table compares departments against fraud classes.

---

## 11. AI Model Performance Center

Open:

**🧠 AI Model Performance Center**

This page shows:

- Model comparison
- Accuracy
- Macro F1
- Feature importance
- Cluster distribution
- Group-based validation
- Confusion matrix
- Fraud precision / recall / F1

### What to focus on

For fraud detection, do not rely on accuracy alone.

Also examine:

- Fraud precision
- Fraud recall
- Fraud F1
- Confusion matrix
- False positives
- False negatives
- Validation on unseen examples

A model that reports 100% accuracy on heavily duplicated training data may still generalize poorly.

---

## 12. Blacklist & Admin Manager

Open:

**⚙️ Blacklist & Admin Manager**

The administrator can add:

- DOMAIN
- COMPANY
- EMAIL
- PHONE
- PAYMENT_HANDLE

Each blacklist entry should include a reason.

Blacklist matches are used as high-severity evidence during verification.

The Review Queue also stores student feedback such as:

- Result Correct
- Result Incorrect
- Report as Scam

Feedback should be reviewed by a human before being added to training data.

---

## 13. Retraining the Model

Retrain whenever the training dataset is substantially updated.

Recommended sequence:

```bash
python scripts/check_dataset.py
python scripts/train_models.py
streamlit run dashboard/streamlit_app.py
```

When changing major library versions, retrain the saved `.joblib` files instead of relying on old model artifacts.

---

## 14. Improving Model Quality

The original demonstration dataset is intentionally synthetic. It should not be treated as enough evidence for real-world accuracy.

For a stronger project:

1. Add thousands of diverse labeled offers.
2. Include real fraudulent and legitimate examples.
3. Add examples from different industries.
4. Add paraphrased scam messages.
5. Include hard negatives: legitimate offers containing words such as “urgent”, “remote”, or “WhatsApp” without actually being fraudulent.
6. Avoid duplicated descriptions across train and validation sets.
7. Keep a completely unseen test set.
8. Tune decision thresholds using fraud precision/recall/F1.
9. Review false positives and false negatives.
10. Retrain only from verified labels.

---

## 15. Important Troubleshooting

### Problem: Every result is 85%

This was caused by the original TrustScore logic:

```python
if not secondary_verification_passed and base_score > 85.0:
    base_score = 85.0
```

That is a display/scoring rule, not a model prediction.

The fixed version removes the hard 85% ceiling.

### Problem: Different inputs still receive very similar predictions

Check:

```bash
python scripts/check_dataset.py
```

A large description-duplication percentage indicates that the model has not learned enough real variation.

Add more diverse training data and retrain.

### Problem: The Performance Center reports extremely high accuracy

Check duplicate descriptions.

Do not trust a 99–100% score when the same examples appear in both training and validation data.

### Problem: Old model results remain after code changes

Retrain:

```bash
python scripts/train_models.py
```

Restart Streamlit afterward.

### Problem: Model loading warnings appear after a Python/scikit-learn upgrade

Retrain the saved model artifacts using the same environment that runs the website.

---

## 16. Final Verification Checklist

Before presenting the project:

- [ ] Dataset has many unique descriptions.
- [ ] Training and test data are separated correctly.
- [ ] Duplicate descriptions are not shared across validation folds.
- [ ] Model probabilities change for different inputs.
- [ ] Trust Score is not hard-coded to 85%.
- [ ] Main website uses the same trained model shown in the Performance Center.
- [ ] Fraud precision, recall, and F1 are recorded.
- [ ] Confusion matrix is available.
- [ ] At least 10 unseen manual test cases were checked.
- [ ] Genuine, suspicious, and fraudulent examples were all tested.
- [ ] Blacklist matching was tested.
- [ ] Payment-demand rules were tested.
- [ ] PDF/image text extraction was tested.
- [ ] Batch CSV scanning was tested.
- [ ] Database records are created correctly.
- [ ] Final recommendation does not claim absolute certainty.

---

## 17. Final User Flow

```text
Job Offer / Email / Offer Letter
              ↓
        Text Extraction
              ↓
      Feature Engineering
              ↓
   ┌──────────┴───────────┐
   ↓                      ↓
 ML Classifier      Rule / Blacklist /
                    Anomaly Evidence
   └──────────┬───────────┘
              ↓
        Combined Risk
           Scoring
              ↓
       Trust Score + Class
              ↓
     Explanation & Alerts
              ↓
       User Decision
```

The safest use of ScamShield AI is as a **decision-support system**: it highlights evidence that deserves attention, while the final verification should still be performed through the employer's official channels.

# 18. Final Real-Data Setup (Recommended)

The final package does not silently use the old repetitive synthetic dataset. Run:

```powershell
python scripts/setup_real_data.py
python scripts/check_dataset.py
python scripts/train_models.py --data data/job_offers.csv
streamlit run dashboard/streamlit_app.py
```

`setup_real_data.py` downloads the EMSCAD-style `fake_job_postings.csv` mirror and
normalizes it into `data/job_offers.csv`. The source dataset has 17,880 labelled job
postings with 866 fraudulent examples in the commonly distributed EMSCAD version;
the labels are binary (genuine/fraudulent). The Suspicious UI state is an uncertainty
state, not a fabricated third training label.

The training script uses a grouped outer holdout based on normalized description so
repeated/template advertisements do not leak across train/test. Model selection
reports fraud precision, fraud recall, fraud F1 and PR-AUC in addition to accuracy.
The selected fraud threshold is saved in `saved_models/best_classifier.joblib`.

After training, the Streamlit Performance Center uses the saved evaluation metrics,
and the Student Verification Portal uses the same combined-feature classifier for
inference.
