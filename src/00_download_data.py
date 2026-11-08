import os
import ssl
import sys
import urllib.error
import urllib.request
import zipfile

"""Download MetroPT-3 from the UCI repository into data/raw/, but only if the
CSV is not already there. Safe to run repeatedly (it is a no-op once the file
exists). Run from anywhere; paths are resolved relative to the repo root.
"""

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CSV_NAME = "MetroPT3(AirCompressor).csv"
CSV_PATH = os.path.join(RAW_DIR, CSV_NAME)
ZIP_PATH = os.path.join(RAW_DIR, "metropt+3+dataset.zip")


def download(req, context):
    with urllib.request.urlopen(req, timeout=120, context=context) as r, open(ZIP_PATH, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main():
    if os.path.exists(CSV_PATH):
        print(f"{CSV_NAME} already present in data/raw/ - nothing to download.")
        return 0
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading {URL} ...")
    # A browser-like User-Agent avoids some servers rejecting urllib's default.
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        try:
            download(req, None)
        except urllib.error.URLError as e:
            if not isinstance(e.reason, ssl.SSLError):
                raise
            # python.org builds on macOS often ship without CA certificates.
            # Retry with certifi's CA bundle (certificate checking stays ON).
            import certifi
            print("System CA certificates unavailable; retrying with certifi.")
            download(req, ssl.create_default_context(cafile=certifi.where()))
    except Exception as e:
        print(f"Download failed: {e}\nDownload manually from "
              "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset "
              f"and put {CSV_NAME} in data/raw/.", file=sys.stderr)
        return 1
    print("Unzipping ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_NAME} not found in the archive.", file=sys.stderr)
        return 1
    print(f"Done: {CSV_PATH} ({os.path.getsize(CSV_PATH) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
