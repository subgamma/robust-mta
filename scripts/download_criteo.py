"""Download the public Criteo Attribution Modeling for Bidding dataset into data/.

    python scripts/download_criteo.py               # downloads data/criteo_attribution_dataset.tsv.gz
    python scripts/download_criteo.py --check-only  # verifies an existing file

Source: https://huggingface.co/datasets/criteo/criteo-attribution-dataset (CC-BY-NC-SA-4.0, see DATA_LICENSE.md).
The download is pinned to dataset revision 904188a63cbad78bee43cd26ff5ee4ac77903986. The file must be
exactly 653,015,824 bytes with SHA-256 94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98.
Uses the standard library and falls back to `huggingface_hub` if installed. If you are offline, download the
file from
    https://huggingface.co/datasets/criteo/criteo-attribution-dataset/blob/main/criteo_attribution_dataset.tsv.gz
and place it at data/criteo_attribution_dataset.tsv.gz.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REVISION = "904188a63cbad78bee43cd26ff5ee4ac77903986"
URL = f"https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/{REVISION}/criteo_attribution_dataset.tsv.gz"
EXPECTED_BYTES = 653_015_824
EXPECTED_SHA256 = "94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98"
MANUAL = (
    "Manual download: open https://huggingface.co/datasets/criteo/criteo-attribution-dataset in a browser, download\n"
    "criteo_attribution_dataset.tsv.gz and save it as data/criteo_attribution_dataset.tsv.gz\n"
    f"(expected size: {EXPECTED_BYTES:,} bytes, SHA-256 {EXPECTED_SHA256}).\n"
    "Then run: python scripts/download_criteo.py --check-only"
)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str) -> bool:
    """Check the exact size first (instant), then the full SHA-256."""
    size = os.path.getsize(path)
    if size != EXPECTED_BYTES:
        print(f"{path}: {size:,} bytes -> WRONG SIZE (expected {EXPECTED_BYTES:,}). The file is truncated or a different "
              "version; delete it and download again (an HTML error page saved under this name also fails here).")
        return False
    digest = sha256_of(path)
    ok = digest == EXPECTED_SHA256
    if ok:
        print(f"{path}: {size:,} bytes, SHA-256 {digest} -> OK")
    else:
        print(f"{path}: size is right but SHA-256 is {digest}, expected {EXPECTED_SHA256}. The file is corrupted or is not "
              f"dataset revision {REVISION}; delete it and download again, or see the manual instructions below.\n{MANUAL}")
    return ok


def download_stdlib(dest: str) -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "anytime-attribution-downloader/1.0"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def download_hf(dest: str) -> None:
    from huggingface_hub import hf_hub_download

    d = os.path.dirname(dest)
    hf_hub_download("criteo/criteo-attribution-dataset", "criteo_attribution_dataset.tsv.gz", repo_type="dataset",
                    revision=REVISION, local_dir=d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data/criteo_attribution_dataset.tsv.gz")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="download again even if a correct file exists")
    a = ap.parse_args()
    if os.path.exists(a.dest) and verify(a.dest) and not a.force:
        return 0
    if a.check_only:
        print(MANUAL)
        return 1
    os.makedirs(os.path.dirname(a.dest) or ".", exist_ok=True)
    try:
        download_stdlib(a.dest)
    except Exception as e:  # offline, blocked, rate-limited
        print(f"Direct download failed ({e}); trying huggingface_hub ...")
        try:
            download_hf(a.dest)
        except Exception as e2:
            print(f"huggingface_hub fallback failed ({e2}).\n{MANUAL}")
            return 1
    return 0 if verify(a.dest) else 1


if __name__ == "__main__":
    sys.exit(main())
