#!/usr/bin/env python3
import os
import json
import pandas as pd
from collections import Counter
from typing import List, Tuple


# ==============================================================
# === EDIT THESE PATHS TO POINT TO YOUR 6 CSV FILES ============
# ==============================================================

fold_csvs = [
    "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/MRI_fold_0.csv",
    "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/MRI_fold_1.csv",
    "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/MRI_fold_2.csv",
    "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/MRI_fold_3.csv",
    "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/MRI_fold_4.csv",
]
test_csv = "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/MRI_fold_test.csv"
#out_json = "/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_preprocessed/Dataset004_AVM_TOFMRA"
out_json = "/hpf/projects/jquon/sumin/labels/folds_TOFMRA/cls_splits_5folds.json"

# ==============================================================


def infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristically infer ID and label columns."""
    cols = list(df.columns)

    id_candidates = [
        "new_name"
    ]
    for c in id_candidates:
        if c in cols:
            id_col = c
            break
    else:
        id_col = next((c for c in cols if df[c].dtype == "object"), None)

    label_candidates = [
        "class_label"
    ]
    for c in label_candidates:
        if c in cols:
            label_col = c
            break
    else:
        label_col = None
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                nunique = df[c].nunique(dropna=True)
                if 1 < nunique <= 10:
                    label_col = c
                    break

    if id_col is None or label_col is None:
        raise ValueError(f"Could not infer id/label columns. Columns present: {cols}")
    return id_col, label_col


def normalize_id(x: str) -> str:
    """Normalize IDs to 'basename.nii.gz'."""
    x = str(x)
    base = os.path.basename(x)
    if base.endswith(".nii.gz"):
        return base
    if "." not in base:
        return base + ".nii.gz"
    return base


def read_fold_csv(path: str, id_col: str = None, label_col: str = None) -> List[Tuple[str,int]]:
    df = pd.read_csv(path)
    if id_col is None or label_col is None:
        id_col, label_col = infer_cols(df)
    return [(normalize_id(r[id_col]), int(r[label_col])) for _, r in df.iterrows()]


def read_test_csv(path: str, id_col: str = None) -> List[str]:
    df = pd.read_csv(path)
    if id_col is None:
        for c in ["new_name"]:
            if c in df.columns:
                id_col = c
                break
        else:
            id_col = next((c for c in df.columns if df[c].dtype == "object"), None)
    return [normalize_id(x) for x in df[id_col].tolist()]


def class_counts(pairs: List[Tuple[str,int]]) -> Counter:
    return Counter([lbl for _, lbl in pairs])


def proportions(cnt: Counter) -> dict:
    total = sum(cnt.values())
    return {k: (v/total if total>0 else 0.0) for k,v in cnt.items()}


def main():
    # Read folds
    fold_items = [read_fold_csv(p) for p in fold_csvs]

    # Read test set
    test_list = read_test_csv(test_csv)

    # Build splits
    splits = {}
    for i in range(5):
        val_pairs = fold_items[i]
        train_pairs = [pair for j, items in enumerate(fold_items) if j != i for pair in items]

        train_ids = sorted(list(set([id_ for (id_, _) in train_pairs]) - set([id_ for (id_, _) in val_pairs])))
        val_ids   = sorted(list(set([id_ for (id_, _) in val_pairs])))

        splits[f"fold_{i}"] = {"train": train_ids, "val": val_ids}

        # Print class balance report
        train_cnt = class_counts(train_pairs)
        val_cnt   = class_counts(val_pairs)
        print(f"\n=== Fold {i} ===")
        print(f"n_train: {len(train_ids)} | n_val: {len(val_ids)}")
        print("train_counts:", dict(train_cnt))
        print("val_counts:  ", dict(val_cnt))
        print("train_props: ", {k: f"{v:.3f}" for k,v in sorted(proportions(train_cnt).items())})
        print("val_props:   ", {k: f"{v:.3f}" for k,v in sorted(proportions(val_cnt).items())})

    splits["test"] = sorted(list(set(test_list)))

    with open(out_json, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
