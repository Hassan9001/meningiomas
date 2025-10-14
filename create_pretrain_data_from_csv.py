#!/usr/bin/env python3
import argparse, os, json, re
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import pandas as pd
# nnssl_preprocess -d 3 -c onemmiso -np 8
# python create_pretrain_data_from_csv.py   --root /hpf/projects/jquon/data/organized_data   --csv /hpf/projects/jquon/sumin/labels/df_combined_MRI_T1.csv   --out /hpf/projects/jquon/sumin/nnssl_dataset/nnssl_raw/Dataset002_AVM_T1/pretrain_data.json   --collection-name Dataset002_AVM_T1   --dataset-id Dataset002_AVM_T1   --session-id ses-DEFAULT   --modality MRI
# python create_pretrain_data_from_csv.py   --root /hpf/projects/jquon/data/organized_data   --csv /hpf/projects/jquon/sumin/labels/concatenate_folds_T1+C.csv   --out /hpf/projects/jquon/sumin/nnssl_dataset/nnssl_raw/Dataset003_AVM_T1+C/pretrain_data.json   --collection-name Dataset003_AVM_T1+C   --dataset-id Dataset003_AVM_T1+C   --session-id ses-DEFAULT   --modality MRI
# python create_pretrain_data_from_csv.py   --root /hpf/projects/jquon/data/organized_data   --csv /hpf/projects/jquon/sumin/labels/concatenate_folds_TOFMRA.csv   --out /hpf/projects/jquon/sumin/nnssl_dataset/nnssl_raw/Dataset004_AVM_TOFMRA/pretrain_data.json   --collection-name Dataset004_AVM_TOFMRA   --dataset-id Dataset004_AVM_TOFMRA   --session-id ses-DEFAULT   --modality MRI
# python create_pretrain_data_from_csv.py   --root /hpf/projects/jquon/data/organized_data   --csv /hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/combined_T1+C_TOFMRA_5folds.csv   --out /hpf/projects/jquon/sumin/nnssl_dataset/nnssl_raw/Dataset005_AVM_T1+C_TOFMRA/pretrain_data.json   --collection-name Dataset005_AVM_T1+C_TOFMRA   --dataset-id Dataset005_AVM_T1+C_TOFMRA   --session-id ses-DEFAULT   --modality MRI
# python /hpf/projects/jquon/models/nnssl/create_pretrain_data_from_csv.py   --root /hpf/projects/jquon/data/organized_data   --csv /hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/combined_T1+C_TOFMRA_5folds.csv   --out /hpf/projects/jquon/sumin/nnssl_dataset/nnssl_raw/Dataset005_AVM_T1+C_TOFMRA/pretrain_data.json   --collection-name Dataset005_AVM_T1+C_TOFMRA   --dataset-id Dataset005_AVM_T1+C_TOFMRA   --session-id ses-DEFAULT   --modality MRI
# python /home/jma/Documents/rsna/create_pretrain_data_from_csv.py   --root /home/jma/Documents/rsna/data/niftiData   --csv train_with_nifti_remaining.csv   --out /home/jma/Documents/rsna/nnssl_dataset/nnssl_raw/Dataset007_RSNA/pretrain_data.json   --collection-name Dataset007_RSNA   --dataset-id Dataset007_RSNA   --session-id ses-DEFAULT   --modality mixed

SUB_BIDS = re.compile(r"(sub-[A-Za-z0-9]+)")

def infer_subject_id(p: Path) -> str:
    m = SUB_BIDS.search(str(p))
    if m: 
        return m.group(1)
    stem = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
    return stem.split("_")[0] if "_" in stem else (p.parent.name or stem)

def maybe_envify(abs_path: str, use_env_paths: bool) -> str:
    if not use_env_paths:  # write absolute path
        return os.path.abspath(abs_path)
    root = os.environ.get("nnssl_raw")
    if root:
        root_abs = os.path.abspath(root)
        ap = os.path.abspath(abs_path)
        if ap == root_abs: return "$nnssl_raw"
        if ap.startswith(root_abs.rstrip(os.sep) + os.sep):
            rel = os.path.relpath(ap, root_abs).replace(os.sep, "/")
            return f"$nnssl_raw/{rel}"
    return os.path.abspath(abs_path)

def build_json(files: List[Path], collection_index: int, collection_name: str,
               dataset_id: str, session_id: str, modality: str,
               use_env_paths: bool) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "collection_index": int(collection_index),
        "collection_name": collection_name,
        "datasets": {
            dataset_id: {
                "dataset_index": dataset_id,
                "dataset_info": {},
                "name": None,
                "subjects": {}
            }
        }
    }
    subjects = data["datasets"][dataset_id]["subjects"]

    for p in tqdm(files, desc="Indexing files"):
        subj = infer_subject_id(p)
        subj_dict = subjects.setdefault(subj, {
            "subject_id": subj, "subject_info": {}, "sessions": {}
        })
        sess = subj_dict["sessions"].setdefault(session_id, {
            "session_id": session_id, "session_info": None, "images": []
        })
        sess["images"].append({
            "associated_masks": {"anatomy_mask": None, "anonymization_mask": None},
            "image_info": {},
            "image_path": maybe_envify(str(p), use_env_paths),
            "modality": modality,
            "name": p.name
        })
    return data

def main():
    ap = argparse.ArgumentParser("Create nnSSL pretrain_data.json from CSV")
    ap.add_argument("--root", required=True, type=Path, help="Directory containing files")
    ap.add_argument("--csv", required=True, type=Path, help="CSV with a 'renamed_nifti_file' column")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--collection-index", type=int, default=1)
    ap.add_argument("--collection-name", type=str, default="Dataset001_Custom")
    ap.add_argument("--dataset-id", type=str, default="Dataset001_Custom")
    ap.add_argument("--session-id", type=str, default="ses-DEFAULT")
    ap.add_argument("--modality", type=str, default="MRI")
    ap.add_argument("--use-env-paths", action="store_true",
                    help="Write $nnssl_raw/... instead of absolute paths")
    args = ap.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv)
    if "renamed_nifti_file" not in df.columns:
        raise SystemExit("CSV must contain a 'renamed_nifti_file' column")

    # Match filenames from CSV against the root folder
    files = []
    for fname in df["renamed_nifti_file"]:
        fpath = args.root / fname
        if not fpath.exists():
            print(f"WARNING: File not found: {fpath}")
            continue
        files.append(fpath)

    if not files:
        raise SystemExit("No files found from CSV renamed_nifti_file column.")

    data = build_json(files, args.collection_index, args.collection_name,
                      args.dataset_id, args.session_id, args.modality,
                      args.use_env_paths)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(data, f, indent=2)

    total = sum(len(s["images"]) for subj in data["datasets"][args.dataset_id]["subjects"].values()
                for s in subj["sessions"].values())
    print(f"Wrote {args.out} with {total} images.")

if __name__ == "__main__":
    main()
