import tarfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

with tarfile.open(DATA_DIR / "raw" / "set-a.tar.gz", "r:gz") as archive:
    for member in archive:
        if member.isfile() and member.name.endswith(".txt"):
            print("Record:", member.name)
            handle = archive.extractfile(member)
            for _ in range(30):
                print(handle.readline().decode("utf-8").strip())
            break