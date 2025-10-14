import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO
import os


def _load_split_list(split_file: Path, split: str, fold=None):
    """
    Supports:
      A) {"train": [...], "val": [...], "test": [...]}
      B) {"fold_0": {"train": [...], "val": [...]}, "fold_1": {...}, ..., "test": [...]}
      C) {"folds": [ {"train": [...], "val": [...]}, ... ], "test": [...]}
      D) [ {"train": [...], "val": [...]}, {"train": [...], "val": [...]}, ... ]  # folds as list

    If fold is None and a fold-based schema is detected, defaults to fold 0.
    """
    import json
    with open(split_file) as f:
        splits = json.load(f)

    # --- test split: prefer top-level "test", else empty list ---
    if split == "test":
        if isinstance(splits, dict) and "test" in splits:
            return splits["test"]
        return []

    # --- no-fold dict schema (A) ---
    if isinstance(splits, dict) and "train" in splits and "val" in splits:
        return splits[split]

    # figure out an index to use if folds exist
    idx = 0 if fold is None else int(fold)

    # --- fold_k dict schema (B) ---
    if isinstance(splits, dict):
        # any fold_* keys?
        has_fold_keys = any(isinstance(k, str) and k.startswith("fold_") for k in splits.keys())
        if has_fold_keys:
            key = f"fold_{idx}"
            if key not in splits:
                raise ValueError(f"{split_file} has fold_* keys but not '{key}'")
            return splits[key][split]

        # --- "folds": [ ... ] schema (C) ---
        if "folds" in splits and isinstance(splits["folds"], list):
            if idx < 0 or idx >= len(splits["folds"]):
                raise IndexError(f"Requested fold {idx} but 'folds' has length {len(splits['folds'])}")
            return splits["folds"][idx][split]

    # --- list of folds schema (D) ---
    if isinstance(splits, list):
        if idx < 0 or idx >= len(splits):
            raise IndexError(f"Requested fold {idx} but split list has length {len(splits)}")
        return splits[idx][split]

    raise ValueError(f"Unsupported cls_splits.json structure in: {split_file}")


class RSNA_Data(Dataset):
    def __init__(self, root, split, fold=None, transform=None):
        super().__init__()
        self.split = split  # "train" | "val" | "test"
        self.img_dir = Path(root) / "nnsslPlans_onemmiso/Dataset007_RSNA/Dataset007_RSNA"
        label_file = Path(root) / "cls_labelsTr.json"
        split_file = Path(root) / "cls_splits_5folds.json"

        # fold can be None now
        self.img_files = _load_split_list(split_file, split=split, fold=fold)

        # labels (robust to missing)
        labels = {}
        if label_file.exists():
            with open(label_file) as f:
                labels = json.load(f)

        lbls, missing = [], 0
        for i in self.img_files:
            if i in labels:
                lbls.append(int(labels[i]))
            else:
                lbls.append(-1)
                missing += 1
        if self.split in ("train", "val") and missing > 0:
            print(f"[WARN] {missing} items in {self.split} split have no labels; set to -1.")
        self.labels = torch.tensor(lbls, dtype=torch.long)

        self.transform = transform

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        subject_id = fname.split("_")[0]
        img_name = fname.replace(".nii.gz", "")
        b2_path = os.path.join(self.img_dir, subject_id, "ses-DEFAULT", img_name + ".b2nd")
        img, _ = Blosc2IO.load(b2_path, mode="r")
        img = torch.from_numpy(img[...])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx], subject_id

    def __len__(self):
        return len(self.img_files)

class RSNA_DataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)

    def setup(self, stage: str):
        # self.fold might be missing or None in config
        fold = getattr(self, "fold", None)

        self.train_dataset = RSNA_Data(
            self.data_path, split="train",
            transform=self.train_transforms, fold=fold,
        )
        self.val_dataset = RSNA_Data(
            self.data_path, split="val",
            transform=self.test_transforms, fold=fold,
        )
        self.test_dataset = RSNA_Data(
            self.data_path, split="test",
            transform=self.test_transforms, fold=fold,
        )