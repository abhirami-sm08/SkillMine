"""Normalize EMSCAD / fake-job-posting CSVs into ScamShield's schema."""
import argparse
import os
import re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_COLUMNS = [
    "title", "company_name", "company_domain", "contact_email", "contact_phone",
    "department", "location", "stipend", "exp_years", "description", "label",
    "employment_type", "required_experience", "required_education", "industry", "function",
    "source",
]


def _first_match(pattern, text):
    match = re.search(pattern, str(text), flags=re.I)
    return match.group(0) if match else ""


def extract_email(text):
    return _first_match(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text)


def extract_phone(text):
    return _first_match(r"(?:\+?\d[\d ()-]{7,}\d)", text)


def extract_domain(text):
    url = _first_match(r"https?://(?:www\.)?[^\s)]+", text)
    if not url:
        return ""
    domain = re.sub(r"^https?://", "", url, flags=re.I).split("/")[0]
    return domain.lower()


def parse_salary_midpoint(value):
    if pd.isna(value):
        return 0.0
    text = str(value).replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return 0.0
    midpoint = float(sum(nums) / len(nums))
    if "month" in text.lower() or "monthly" in text.lower():
        return midpoint
    # EMSCAD salary ranges are generally annual USD amounts.
    if "year" in text.lower() or "annual" in text.lower() or midpoint >= 5000:
        return midpoint / 12.0
    return midpoint


def parse_experience(value):
    text = str(value).lower().strip()
    if not text or text == "nan":
        return 0.0
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[0])
    mapping = {
        "internship": 0.0,
        "entry level": 0.0,
        "associate": 1.0,
        "mid-senior level": 3.0,
        "director": 6.0,
        "executive": 7.0,
        "not applicable": 0.0,
    }
    for key, value_out in mapping.items():
        if key in text:
            return value_out
    return 0.0


def prepare(input_path, output_path):
    df = pd.read_csv(input_path, low_memory=False)

    text_cols = [
        c for c in ["title", "company_profile", "description", "requirements", "benefits"]
        if c in df.columns
    ]
    if not text_cols:
        raise ValueError("No supported job-posting text columns found.")

    full_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

    out = pd.DataFrame(index=df.index)
    out["title"] = df["title"].fillna("").astype(str) if "title" in df else ""
    # EMSCAD does not provide a clean company-name field. Keep a neutral label
    # instead of pretending a sentence from company_profile is the company name.
    out["company_name"] = "Not provided"
    out["company_domain"] = full_text.map(extract_domain)
    out["contact_email"] = full_text.map(extract_email)
    out["contact_phone"] = full_text.map(extract_phone)
    out["department"] = (
        df["department"].fillna("").astype(str)
        if "department" in df.columns
        else df["function"].fillna("").astype(str) if "function" in df.columns else "General"
    )
    out["location"] = df["location"].fillna("").astype(str) if "location" in df.columns else ""
    out["stipend"] = df["salary_range"].apply(parse_salary_midpoint).astype(float) if "salary_range" in df.columns else 0.0
    out["exp_years"] = df["required_experience"].apply(parse_experience).astype(float) if "required_experience" in df.columns else 0.0
    out["description"] = full_text

    if "fraudulent" in df.columns:
        fraud = pd.to_numeric(df["fraudulent"], errors="coerce")
        out["label"] = np.where(fraud.fillna(0).astype(int).eq(1), "Fraudulent", "Genuine")
    elif "label" in df.columns:
        out["label"] = df["label"].astype(str)
    else:
        raise ValueError("Dataset must contain `fraudulent` or `label`.")

    for source_col in ["employment_type", "required_experience", "required_education", "industry", "function"]:
        out[source_col] = df[source_col].fillna("").astype(str) if source_col in df.columns else ""
    out["source"] = "EMSCAD/public mirror"

    out = out[OUTPUT_COLUMNS]
    # Exact duplicate ads are removed; repeated-but-different records remain.
    out["_norm_desc"] = out["description"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.drop_duplicates().drop_duplicates(subset=["_norm_desc"]).drop(columns=["_norm_desc"]).reset_index(drop=True)

    if out["label"].nunique() < 2:
        raise ValueError("The prepared dataset must contain at least two classes.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[OK] Saved {len(out):,} rows to {output_path}")
    print("\nLabel distribution:")
    print(out["label"].value_counts())
    print(f"\nUnique descriptions: {out['description'].nunique():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default=os.path.join(ROOT, "data", "job_offers.csv"))
    args = parser.parse_args()
    prepare(args.input_csv, args.output)
