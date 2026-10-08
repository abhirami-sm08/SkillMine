# ScamShield AI — Model Remediation Notes

## Root causes found

1. `TrustScoreEngine` hard-capped any score above 85%, so a clean offer could always display exactly 85%.
2. The website predicted with `model_pipeline`, a text-only Logistic Regression pipeline, while the Performance Center reported `best_classifier.joblib`, which combines TF-IDF with 12 structured/risk features. The website and the reported classifier were therefore not the same inference path.
3. The training CSV contains 450 rows but only 10 unique descriptions. About 97.8% of rows reuse an identical description.
4. The old saved classifier reports 100% holdout accuracy/F1, but that result is inflated by duplicated templates.
5. The training script previously dropped rows by `description`, which can collapse a repetitive dataset to only a few unique examples.
6. `TextCleaner._init_nlp()` could return before NLTK stopwords were initialized, producing a `NoneType` error in some environments.
7. `LogisticRegression(multi_class="auto")` is incompatible with newer scikit-learn releases; the flag is unnecessary for multinomial-capable LogisticRegression defaults.
8. The old probability cap clipped model probabilities at 0.85 and renormalized them, changing the model's actual probabilities.
9. The original holdout evaluation fit TF-IDF before the train/test split, allowing test vocabulary statistics into the feature extractor.

## What the fixed starter build changes

- Removes the artificial 85% Trust Score ceiling.
- Uses the combined feature classifier for website inference.
- Displays raw model class probabilities.
- Assigns `Suspicious` as an operational uncertainty state when class confidence is weak or the probability margin is small.
- Makes dataset health visible with `scripts/check_dataset.py`.
- Adds `scripts/prepare_real_dataset.py` for EMSCAD-style real/fake job-posting data.
- Lets `scripts/train_models.py` prefer `data/job_offers.csv`.
- Accepts binary `fraudulent` targets and maps them to Genuine/Fraudulent.
- Fits the evaluation feature extractor on the training split only.
- Refits the selected model on the full dataset after evaluation.
- Makes GroupKFold duplicate-aware by grouping repeated descriptions.
- Fixes the NLP initialization fallback and scikit-learn compatibility issue.
- Adds a visible ML probability distribution chart.
- Updates the user manual so it does not promise fixed/demo scores or absolute fraud certainty.

## What still must be done for a strong final project

The included synthetic dataset is still not a production-quality fraud dataset. Replace it with a diverse, labeled real-world dataset and then retrain all `.joblib` artifacts.

Do not claim "correct every time". Instead report:
- fraud precision
- fraud recall
- fraud F1
- confusion matrix
- validation methodology
- unseen manual test results

A strong final demo should include genuinely unseen offers and hard-negative examples.
