# Real labelled dataset

ScamShield is configured to train on the Employment Scam Aegean Dataset (EMSCAD),
commonly distributed as the Kaggle `fake_job_postings.csv` dataset.

The project does **not** bundle the ~50 MB source CSV. Run from the project root:

```powershell
python scripts/setup_real_data.py
python scripts/check_dataset.py
python scripts/train_models.py --data data/job_offers.csv
```

The downloader uses a public Hugging Face mirror of the dataset:
https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction

The dataset is a **binary** classification task: `fraudulent = 0` is genuine and
`fraudulent = 1` is fraudulent. ScamShield derives its third UI state, `Suspicious`,
from model uncertainty and deterministic risk evidence instead of inventing labels.

For academic attribution, cite the original EMSCAD publication by Vidros et al.
and the dataset source used for your final submission.
