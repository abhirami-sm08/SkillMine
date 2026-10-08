"""Quality gate for ScamShield training data."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    path = os.path.join(ROOT, "data", "job_offers.csv")
    if not os.path.exists(path):
        path = None
    if not path:
        raise FileNotFoundError("No training CSV found. Run `python scripts/setup_real_data.py`.")

    df = pd.read_csv(path, low_memory=False)
    label_col = "label" if "label" in df.columns else "fraudulent" if "fraudulent" in df.columns else None
    if "description" not in df.columns and not any(c in df.columns for c in ["title", "company_profile", "requirements", "benefits"]):
        raise ValueError("No supported job-posting text columns were found.")
    if not label_col:
        raise ValueError("No `label` or `fraudulent` target column found.")

    desc = df.get("description", pd.Series([""] * len(df))).fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    rows = len(df)
    unique = int(desc.nunique())
    dup_ratio = 1.0 - unique / max(1, rows)

    print(f"Dataset: {path}")
    print(f"Rows: {rows:,}")
    print(f"Unique normalized descriptions: {unique:,}")
    print(f"Description duplication: {dup_ratio*100:.2f}%")
    print("\nLabel distribution:")
    print(df[label_col].value_counts(dropna=False))

    if rows < 2000:
        print("\nWARNING: Dataset is small for fraud detection; use the real EMSCAD-sized dataset for final evaluation.")
    if dup_ratio > 0.30:
        print("WARNING: High duplicate-description rate. Grouped validation is required and generalization may be weak.")
    if label_col == "fraudulent":
        values = pd.to_numeric(df[label_col], errors="coerce")
        fraud_n = int((values == 1).sum())
        print(f"Fraudulent rows: {fraud_n:,}")
        if fraud_n < 50:
            print("WARNING: Too few fraud examples for reliable recall estimates.")


if __name__ == "__main__":
    main()
