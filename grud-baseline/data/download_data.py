from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

# Resolve data and results relative to grud-baseline, regardless of the terminal folder.
BASE_DIR = Path(__file__).resolve().parent.parent
from urllib.request import urlretrieve

BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0/"
FILES = {
    "set-a.tar.gz": "8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920",
    "Outcomes-a.txt": "2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.output is not None and not args.output.is_absolute():
        args.output = BASE_DIR / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = args.output / name
        if not path.exists():
            print(f"Downloading {name}...")
            urlretrieve(BASE_URL + name, path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual}")
        print(f"Verified {path} ({path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()