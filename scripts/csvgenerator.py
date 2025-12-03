import os, re, csv, math, random
from pathlib import Path

ROOT = Path("dataset")
LABEL_DIRS = ["AES","ASCONv2","LEA","MSEA","PRESENT","PRINCE","RECTANGLE","SIMON","XTEA"]
OUT_CSV = ROOT/"dataset_manifest.csv"
SEED = 42
SPLIT = (0.8, 0.1, 0.1)  # train/val/test

random.seed(SEED)

def parse_size_from_name(name):
    # tries to read patterns like "..._16KB_..." or "16KB_0.bin"
    m = re.search(r"(\d+)\s*KB", name.upper())
    return int(m.group(1)) if m else None

rows = []
for label in LABEL_DIRS:
    d = ROOT/label
    if not d.is_dir():
        continue
    for p in d.glob("*.bin"):
        size_bytes = p.stat().st_size
        size_kb_actual = math.ceil(size_bytes/1024)
        size_kb_name = parse_size_from_name(p.name)
        rows.append({
            "path": str(p.as_posix()),
            "label": label,
            "filename": p.name,
            "size_bytes": size_bytes,
            # prefer size from name if present, else actual
            "size_kb": size_kb_name if size_kb_name is not None else size_kb_actual,
            # ASCON includes 16-byte tag at end (for features you may strip it)
            "has_tag": 1 if label.upper()=="ASCONV2" else 0
        })

# stratified split by label (and roughly by size_kb buckets)
def stratified_split(items):
    by_label = {}
    for r in items:
        key = (r["label"], r["size_kb"])
        by_label.setdefault(key, []).append(r)

    splits = []
    for key, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n*SPLIT[0])
        n_val   = int(n*SPLIT[1])
        for i, r in enumerate(group):
            r2 = dict(r)
            r2["split"] = "train" if i < n_train else ("val" if i < n_train+n_val else "test")
            splits.append(r2)
    return splits

rows = stratified_split(rows)
fields = ["path","label","filename","size_bytes","size_kb","has_tag","split"]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"wrote {OUT_CSV} with {len(rows)} rows")
