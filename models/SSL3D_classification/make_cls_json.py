#!/usr/bin/env python3
import csv, json
from pathlib import Path

# ===================== CONFIG (edit these) =====================
TRAIN_CSV = "/hpf/projects/jquon/sumin/labels/train_split_MRI_T1.csv"
VAL_CSV   = "/hpf/projects/jquon/sumin/labels/val_split_MRI_T1.csv"
TEST_CSV  = "/hpf/projects/jquon/sumin/labels/test_split_MRI_T1.csv"
OUTDIR    = "/hpf/projects/jquon/models/SSL3D_classification"  # or "/path/to/output/folder" (defaults to parent of TRAIN_CSV if None)

ID_COL    = "new_name"
LABEL_COL = "class_label"
# ===============================================================

def read_csv(path, id_col="new_name", y_col="class_label"):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        ids, labels = [], {}
        for row in r:
            rid = str(row[id_col]).strip()
            if not rid:
                continue
            ids.append(rid)
            # label may exist in test too; coerce to int if possible
            y = row.get(y_col, "")
            try:
                labels[rid] = int(str(y).strip())
            except Exception:
                labels[rid] = str(y).strip()
    # de-dup preserving order
    seen, ids_unique = set(), []
    for x in ids:
        if x not in seen:
            ids_unique.append(x); seen.add(x)
    return ids_unique, labels

def main():
    train_p, val_p, test_p = Path(TRAIN_CSV), Path(VAL_CSV), Path(TEST_CSV)
    outdir = Path(OUTDIR) if OUTDIR else train_p.parent
    outdir.mkdir(parents=True, exist_ok=True)

    tr_ids, tr_labels = read_csv(train_p, ID_COL, LABEL_COL)
    va_ids, va_labels = read_csv(val_p,   ID_COL, LABEL_COL)
    te_ids, te_labels = read_csv(test_p,  ID_COL, LABEL_COL)

    # splits
    splits = {"train": tr_ids, "val": va_ids, "test": te_ids}
    with open(outdir / "cls_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # labelsTr: include train + val + test
    labels_all = {}
    for d in (tr_labels, va_labels, te_labels):
        labels_all.update(d)
    with open(outdir / "cls_labelsTr.json", "w") as f:
        json.dump(labels_all, f, indent=2)

    # small summary
    print(f"[OK] wrote {outdir/'cls_splits.json'}  (train={len(tr_ids)}, val={len(va_ids)}, test={len(te_ids)})")
    print(f"[OK] wrote {outdir/'cls_labelsTr.json'} (labels={len(labels_all)})")

    # sanity: warn about overlaps
    s_tr, s_va, s_te = set(tr_ids), set(va_ids), set(te_ids)
    for name, inter in [("train∩val", s_tr & s_va), ("train∩test", s_tr & s_te), ("val∩test", s_va & s_te)]:
        if inter:
            print(f"[WARN] overlap in {name}: {len(inter)} items")

if __name__ == "__main__":
    main()
