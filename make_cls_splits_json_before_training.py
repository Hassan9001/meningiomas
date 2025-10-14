#!/usr/bin/env python3
import os
import json
import pandas as pd
from collections import Counter

# ==============================================================
# === EDIT THESE PATHS TO POINT TO YOUR NEW CSV FILE ============
# ==============================================================

# in_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/MRI_T1+C_5folds.csv"
# in_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/MRI_TOFMRA_5folds.csv"
in_csv = "/home/jma/Documents/rsna/train_with_nifti_remaining.csv"
out_json = "/home/jma/Documents/rsna/cls_splits_5folds.json"

# =============================================================


def normalize_id(x: str) -> str:
    """Normalize IDs to 'basename.nii.gz'."""
    x = str(x)
    base = os.path.basename(x)
    if base.endswith(".nii.gz"):
        return base
    if "." not in base:
        return base + ".nii.gz"
    return base


def class_counts(labels):
    return Counter(labels)


def proportions(cnt: Counter) -> dict:
    total = sum(cnt.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in cnt.items()}


def main():
    df = pd.read_csv(in_csv)

    # Normalize file IDs
    df["renamed_nifti_file"] = df["renamed_nifti_file"].apply(normalize_id)

    splits = {}

    # Build 5 folds
    for i in range(5):
        val_df = df[df["fold"] == i]
        train_df = df[df["fold"] != i]

        val_ids = sorted(val_df["renamed_nifti_file"].unique().tolist())
        train_ids = sorted(list(set(train_df["renamed_nifti_file"].tolist()) - set(val_ids)))

        splits[f"fold_{i}"] = {"train": train_ids, "val": val_ids}

        # Print class balance report
        train_cnt = class_counts(train_df["Aneurysm Present"].tolist())
        val_cnt = class_counts(val_df["Aneurysm Present"].tolist())
        print(f"\n=== Fold {i} ===")
        print(f"n_train: {len(train_ids)} | n_val: {len(val_ids)}")
        print("train_counts:", dict(train_cnt))
        print("val_counts:  ", dict(val_cnt))
        print("train_props: ", {k: f"{v:.3f}" for k, v in sorted(proportions(train_cnt).items())})
        print("val_props:   ", {k: f"{v:.3f}" for k, v in sorted(proportions(val_cnt).items())})

    # No explicit "test" set in new CSV — add empty list for compatibility
    splits["test"] = []

    with open(out_json, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
