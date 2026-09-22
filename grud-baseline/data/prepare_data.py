from __future__ import annotations
import argparse, csv, json, tarfile
from pathlib import Path

# Resolve data and results relative to grud-baseline, regardless of the terminal folder.
BASE_DIR = Path(__file__).resolve().parent.parent
import numpy as np
from sklearn.model_selection import StratifiedKFold

FEATURES = ["ALP","ALT","AST","Albumin","BUN","Bilirubin","Cholesterol","Creatinine",
            "DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","K","Lactate","MAP",
            "Mg","Na","PaCO2","PaO2","Platelets","RespRate","SaO2","SysABP","Temp",
            "TroponinI","TroponinT","Urine","WBC","Weight","pH"]

MERGE = {"NIDiasABP":"DiasABP", "NIMAP":"MAP", "NISysABP":"SysABP"}

def minutes(clock):
    hour, minute = map(int, clock.split(":"))
    return 60 * hour + minute

def compute_delta(mask, times):
    delta = np.zeros_like(mask, dtype=np.float32)
    for t in range(1, len(times)):
        gap = times[t] - times[t - 1]
        delta[t] = gap + (1 - mask[t - 1]) * delta[t - 1]
    return delta


def read_all(raw):
    labels = {}
    with (raw / "Outcomes-a.txt").open(newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row["RecordID"]
            if record_id in labels:
                raise ValueError(f"Duplicate outcome ID: {record_id}")
            labels[record_id] = int(row["In-hospital_death"])
    if set(labels.values()) != {0, 1}:
        raise ValueError("Mortality labels must be binary")
    feature_index = {name:i for i,name in enumerate(FEATURES)}
    records = []
    with tarfile.open(raw / "set-a.tar.gz", "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            record_id = Path(member.name).stem
            cells = {}
            rows = csv.DictReader(line.decode("utf-8") for line in archive.extractfile(member))
            for row in rows:
                name = MERGE.get(row["Parameter"], row["Parameter"])
                if name not in feature_index:
                    continue
                try:
                    value, minute = float(row["Value"]), minutes(row["Time"])
                except ValueError:
                    continue
                if np.isfinite(value) and value >= 0 and 0 <= minute <= 2880:
                    cells[(minute // 10, feature_index[name])] = value
            records.append((record_id, cells, labels[record_id]))
    record_ids = [item[0] for item in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate measurement record ID")
    if set(record_ids) != set(labels):
        raise ValueError("Measurement and outcome IDs are not a one-to-one match")
    return sorted(records, key=lambda item:item[0])

def tensorize(records):
    sequences = []
    for record_id, cells, label in records:
        occupied = sorted({slot for slot, _ in cells}) or [0]
        x = np.full((len(occupied), len(FEATURES)), np.nan, np.float32)
        for (slot, feature), value in cells.items():
            x[occupied.index(slot), feature] = value
        times = np.asarray(occupied, np.float32) / 6.0
        sequences.append((record_id, x, np.isfinite(x).astype(np.float32), times, label))
    maximum = max(len(item[1]) for item in sequences)
    n, d = len(sequences), len(FEATURES)
    x = np.full((n, maximum, d), np.nan, np.float32)
    mask = np.zeros((n, maximum, d), np.float32)
    delta = np.zeros_like(mask)
    times = np.zeros((n, maximum), np.float32)
    lengths = np.empty(n, np.int64)
    for i, (_, values, observed, clock, _) in enumerate(sequences):
        length = len(values); lengths[i] = length
        x[i,:length], mask[i,:length], times[i,:length] = values, observed, clock
        delta[i,:length] = compute_delta(observed, clock)
    return x, mask, delta, times, lengths

def folds(y):
    fold = np.empty(len(y), np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for number, (_, test_indices) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold[test_indices] = number
    return fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/p12_paper.npz"))
    args = parser.parse_args()
    if args.raw is not None and not args.raw.is_absolute():
        args.raw = BASE_DIR / args.raw
    if args.output is not None and not args.output.is_absolute():
        args.output = BASE_DIR / args.output
    records = read_all(args.raw)
    x, mask, delta, times, lengths = tensorize(records)
    ids = np.asarray([item[0] for item in records])
    y = np.asarray([item[2] for item in records], np.int64)
    x = np.where(mask > 0, x, 0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, mask=mask, delta=delta, times=times,
                        lengths=lengths, ids=ids, y=y, features=np.asarray(FEATURES), fold_test=folds(y))
    empty_count = int(sum(not item[1] for item in records))
    metadata = {"records":len(ids),"features":FEATURES,"maximum_length":int(lengths.max()),
                "mean_length":float(lengths.mean()),"empty_record_count":empty_count}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(metadata)
