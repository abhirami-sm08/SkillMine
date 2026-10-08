"""Download the public EMSCAD fake-job-posting dataset mirror.

The original project should not bundle the ~50 MB CSV. This script downloads it
from a public Hugging Face dataset mirror, after which prepare_real_dataset.py
normalizes it into ScamShield's training schema.
"""
import os
import sys
import urllib.request
import shutil


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(RAW_DIR, "fake_job_postings.csv")
URL = (
    "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/"
    "resolve/main/fake_job_postings.csv?download=true"
)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size and total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\rDownloading EMSCAD: {pct:6.2f}%")
        sys.stdout.flush()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(OUT) and os.path.getsize(OUT) > 10_000_000:
        print(f"[OK] Dataset already exists: {OUT}")
        return

    print("[*] Source: public Hugging Face mirror of EMSCAD / fake job postings")
    print("[*] Download size is approximately 50 MB.")
    try:
        request = urllib.request.Request(
            URL,
            headers={"User-Agent": "ScamShield-AI/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, open(OUT + ".part", "wb") as target:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                downloaded += len(block)
                if total:
                    print(f"\rDownloading EMSCAD: {downloaded * 100 / total:6.2f}%", end="", flush=True)
        os.replace(OUT + ".part", OUT)
        print("\n[OK] Download complete:", OUT)
    except Exception as exc:
        for candidate in (OUT + ".part", OUT):
            if os.path.exists(candidate):
                os.remove(candidate)
        raise RuntimeError(
            "Dataset download failed. Check your internet connection or download "
            "fake_job_postings.csv manually and place it at data/raw/fake_job_postings.csv."
        ) from exc


if __name__ == "__main__":
    main()
