"""One-command download + normalization for the real labelled job-fraud dataset."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from scripts.download_real_dataset import main as download_main
from scripts.prepare_real_dataset import prepare


def main():
    download_main()
    raw = os.path.join(ROOT, "data", "raw", "fake_job_postings.csv")
    output = os.path.join(ROOT, "data", "job_offers.csv")
    prepare(raw, output)
    print("\n[OK] Real labelled dataset is ready.")
    print("[*] Next command: python scripts/check_dataset.py")
    print("[*] Then:        python scripts/train_models.py --data data/job_offers.csv")


if __name__ == "__main__":
    main()
