# ScamShield AI — Final ML Build

This version fixes the constant-85% problem and removes the old synthetic model artifacts.
The app uses a binary-first fraud model (Genuine vs Fraudulent) and derives **Suspicious**
from model uncertainty plus deterministic risk evidence.

## Real labelled data setup

From the project root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/setup_real_data.py
python scripts/check_dataset.py
python scripts/train_models.py --data data/job_offers.csv
python scripts/test_model_examples.py
streamlit run dashboard\streamlit_app.py
```

The downloader retrieves the commonly distributed EMSCAD `fake_job_postings.csv` mirror.
The source contains 17,880 labelled job postings and 866 fraudulent postings in the
standard version, so the task is severely imbalanced and fraud recall/precision/F1 and
PR-AUC should be emphasized over raw accuracy.

## Manual download fallback

Download `fake_job_postings.csv` from the dataset mirror below and place it at:

`data/raw/fake_job_postings.csv`

Then run:

```powershell
python scripts/prepare_real_dataset.py data/raw/fake_job_postings.csv
python scripts/check_dataset.py
python scripts/train_models.py --data data/job_offers.csv
```

Dataset mirror:
https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction

## What was fixed

- Removed the artificial 85% Trust Score ceiling.
- Removed probability clipping/renormalization that changed model output.
- Website inference now uses the same combined-feature classifier as the model center.
- Real EMSCAD-style binary labels are supported directly.
- Suspicious is derived from uncertainty instead of fabricated training labels.
- TF-IDF is fitted on the training split before the grouped held-out test split.
- Final test split is grouped by normalized description to reduce duplicate/template leakage.
- Fraud threshold is tuned on an inner grouped validation split and saved with the model.
- Model comparison reports fraud precision, recall, F1, ROC-AUC and PR-AUC.
- The final package contains no old synthetic model artifacts.

## Important limitation

No ML model can guarantee a correct answer for every unseen job posting. EMSCAD is an
older dataset; current scam tactics can differ. Use the system as decision support and
verify employers independently.
