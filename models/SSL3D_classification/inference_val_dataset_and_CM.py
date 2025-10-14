import os
import re
from pathlib import Path
from uuid import uuid4

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
import pandas as pd

from parsing_utils import make_omegaconf_resolvers
import glob
from metrics.balanced_accuracy import BalancedAccuracy

from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, accuracy_score
)
from sklearn.preprocessing import label_binarize

# ======== HARD-CODED CLASSES (edit to your case) ========
# Display/order you want on the confusion matrix and for prediction indices
CLASS_NAMES = ["0", "1"] # AVM

# What integers are used in your GROUND-TRUTH labels (from cls_labelsTr.json),
# in the SAME order as CLASS_NAMES above.
#   e.g. if GT uses {0,1}  with 0=Control, 1=AVM  -> [0, 1]
#        if GT uses {1,2}  with 1=Control, 2=AVM  -> [1, 2]
#        if GT uses {-1,1} with -1=unlabeled, 1=AVM (and Control is 0) -> [-1, 1]  (we’ll ignore -1)
RAW_TRUE_LABEL_IDS = [0, 1]
# =========================================================


# -------------------- NEW: checkpoint parsing & selection --------------------
_ckpt_rx = re.compile(
    r"epoch(?P<epoch>\d+).*?"
    r"Val_loss=(?P<val_loss>\d+(?:\.\d+)?).*?"
    r"Val_(?:bal|bala|balanced)_acc=(?P<bal_acc>\d+(?:\.\d+)?)\.ckpt",
    re.IGNORECASE
)


#=============================Confusion matrix helper=======================================================# 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#==============Delete this================
# def _get_class_names(cfg_model) -> list:
#     # Try common locations; adjust if you store names elsewhere
#     for key in ("class_names", "classes", "labels"):
#         if key in cfg_model and cfg_model[key]:
#             return list(cfg_model[key])
#     # fallback to indices
#     n = int(cfg_model["num_classes"])
#     return [str(i) for i in range(n)]
#=======================================
def _save_confusion_matrix_assets(y_true, y_pred, class_names, out_prefix: str, fold_title):
    """
    Saves:
      - {out_prefix}_counts.csv
      - {out_prefix}_norm.csv (row-normalized)
      - {out_prefix}.png (heatmap)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with open(out_prefix + "_counts.csv", "w") as f:
        # header
        f.write(",".join([""] + [f"pred_{c}" for c in class_names]) + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([f"true_{class_names[i]}"] + [str(v) for v in row.tolist()]) + "\n")

    # row-normalized (handle divide-by-zero rows)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums != 0)
    with open(out_prefix + "_norm.csv", "w") as f:
        f.write(",".join([""] + [f"pred_{c}" for c in class_names]) + "\n")
        for i, row in enumerate(cm_norm):
            f.write(",".join([f"true_{class_names[i]}"] + [f"{v:.6f}" for v in row.tolist()]) + "\n")

    # # heatmap
    # fig, ax = plt.subplots(figsize=(1.2*len(class_names), 1.0*len(class_names)))
    # im = ax.imshow(cm_norm, interpolation="nearest")
    # ax.set_title("Confusion Matrix (row-normalized)")
    # ax.set_xlabel("Predicted")
    # ax.set_ylabel("True")
    # ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    # ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    # # annotate cells
    # for i in range(cm_norm.shape[0]):
    #     for j in range(cm_norm.shape[1]):
    #         ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # fig.tight_layout()
    # fig.savefig(out_prefix + ".png", dpi=200)
    # plt.close(fig)
    # heatmap (green = best, dark purple = worst)
    fig, ax = plt.subplots(figsize=(1.2*len(class_names), 1.0*len(class_names)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="PRGn", vmin=0.0, vmax=1.0)

    ax.set_title(f"Fold-{fold_title}", fontsize=11)#(f"{'F'+(fold_title[1:])}", fontsize=10) #Confusion Matrix (norm)", fontsize=10) # ROW-NORMALIZED
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, ha="right", fontsize=8) #rotation=45
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)

    # annotate cells with readable text color
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            txt_color = "white" if val >= 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=200)
    plt.close(fig)

### THESE ONEs IS FOR CONFIDINCE INTERVALS
def _compute_metrics_numpy(y_true_idx, y_pred_idx, y_score, num_classes):
    """
    y_true_idx: (N,) integers in [0..C-1]
    y_pred_idx: (N,) integers in [0..C-1]
    y_score:    (N,C) logits or probabilities
    Returns point estimates for Accuracy, Balanced Acc, F1(macro), AUROC(macro), AP(macro)
    """
    # Accuracy
    acc = accuracy_score(y_true_idx, y_pred_idx)
    # Balanced Acc (avg of per-class recall)
    bal_acc = balanced_accuracy_score(y_true_idx, y_pred_idx)
    # F1 (macro)
    f1m = f1_score(y_true_idx, y_pred_idx, average="macro")

    # For AUROC/AP (macro, one-vs-rest), binarize y_true
    # (sklearn can do multiclass AUROC/AP directly if using v1.3+, but this is explicit & robust)
    Y = label_binarize(y_true_idx, classes=list(range(num_classes)))
    # Convert logits -> probabilities via softmax for AP (AUROC works with scores, but softmax OK)
    # Avoid overflow
    s = y_score - y_score.max(axis=1, keepdims=True)
    prob = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)

    # AUROC macro
    try:
        auroc = roc_auc_score(Y, prob, average="macro", multi_class="ovr")
    except ValueError:
        # If a fold is missing one class entirely, AUROC is undefined -> set NaN
        auroc = float("nan")

    # Average Precision (macro)
    try:
        ap = average_precision_score(Y, prob, average="macro")
    except ValueError:
        ap = float("nan")

    return {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "F1 Score": f1m,
        "AUROC": auroc,
        "Average Precision": ap
    }


def _bootstrap_ci(y_true_idx, y_pred_idx, y_score, num_classes, B=2000, seed=123):
    """
    Nonparametric bootstrap (patient-level). Returns dict of (low, high) per metric.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_idx)
    accs, bals, f1s, aucs, aps = [], [], [], [], []

    indices = np.arange(n)
    for _ in range(B):
        bs_idx = rng.integers(0, n, size=n)           # sample with replacement
        yt = y_true_idx[bs_idx]
        yp = y_pred_idx[bs_idx]
        ys = y_score[bs_idx]

        m = _compute_metrics_numpy(yt, yp, ys, num_classes)
        accs.append(m["Accuracy"])
        bals.append(m["Balanced Accuracy"])
        f1s.append(m["F1 Score"])
        aucs.append(m["AUROC"])
        aps.append(m["Average Precision"])

    def pct(vals):
        vals = np.array(vals, dtype=float)
        return (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))

    return {
        "Accuracy": pct(accs),
        "Balanced Accuracy": pct(bals),
        "F1 Score": pct(f1s),
        "AUROC": pct(aucs),
        "Average Precision": pct(aps),
    }
#====================================================================================



def _parse_ckpt_metrics(p: Path):
    """
    Returns (bal_acc: float, val_loss: float, epoch: int) parsed from the filename.
    Raises ValueError if it doesn't match the expected pattern.
    """
    m = _ckpt_rx.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse metrics from checkpoint name: {p.name}")
    epoch = int(m.group("epoch"))
    val_loss = float(m.group("val_loss"))
    bal_acc = float(m.group("bal_acc"))
    return bal_acc, val_loss, epoch

def select_best_checkpoint(ckpt_paths):
    """
    Select best checkpoint by:
      1) max bal_acc
      2) min val_loss
      3) max epoch
    Returns a single Path.
    """
    parsed = []
    for p in ckpt_paths:
        try:
            bal_acc, val_loss, epoch = _parse_ckpt_metrics(p)
            parsed.append((bal_acc, val_loss, epoch, p))
        except ValueError:
            # skip files that don't match pattern
            continue

    if not parsed:
        raise RuntimeError("No checkpoints with parseable metrics were found.")

    # Sort using our tie-break rule:
    # -bal_acc (desc), +val_loss (asc), -epoch (desc)
    parsed.sort(key=lambda x: (-x[0], x[1], -x[2]))
    best = parsed[0]
    best_bal, best_loss, best_epoch, best_path = best
    print(
        f"[ckpt-select] Selected: {best_path}\n"
        f"  -> Val_bal_acc={best_bal:.6f}, Val_loss={best_loss:.6f}, epoch={best_epoch}"
    )
    return best_path
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer_cls_val")
def inference(cfg):
    
    #--SET-PATH------------------------------------------------------------------------
    # 5 fold v2 split
    #cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset002_AVM_T1/ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-3_5folds/AVM_T1/checkpoints'
    #cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset002_AVM_T1/ResEnc_no_pretrained_cls_DblWarmup_lr_1e-3/AVM_T1/checkpoints'
    # 5 fold T1+C and TOFMRA
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/balanced_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-3_5folds_bsize8/AVM_T1+C/checkpoints'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/scratch_balanced_lr_1e-3_5folds_bsize8/AVM_T1+C/checkpoints'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/v2_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds/AVM_T1+C/checkpoints'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/scratch_balanced_lr_1e-3_5folds_bsize8'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/balanced_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_bsize8/AVM_T1+C/checkpoints'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/NoHoldOut_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_balanced_bsize8/AVM_T1+C/checkpoints'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset004_AVM_TOFMRA/NoHoldOut_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_balanced_bsize8/AVM_TOFMRA/checkpoints'
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset005_AVM_T1+C_TOFMRA/NoHoldOut_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_balanced_bsize8/AVM_T1+C_TOFMRA/checkpoints'




    #SCRATCH TOFMRA AND T1C
    # cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/NoHoldOut_scratch_cls_DblWarmup_lr_1e-4_5folds_balanced_bsize8/AVM_T1+C/checkpoints'
    cfg.exp_dir = '/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset004_AVM_TOFMRA/NoHoldOut_scratch_cls_DblWarmup_lr_1e-4_5folds_balanced_bsize8/AVM_TOFMRA/checkpoints'




    
    print(f'performing inference on {cfg.exp_dir}')
    #----------------------------------------------------------------------------------

    # delete automatically created hydra logger
    try:
        Path("./main.log").unlink()
    except:
        pass

    # collect candidate checkpoints
    if cfg.fold:
        #ckpt_paths_all = list(Path(Path(cfg.ckpt_dir) / str(cfg.fold)).glob("*.ckpt"))
        ckpt_paths_all = glob.glob(os.path.join(cfg.exp_dir, '*', str(cfg.fold), "*.ckpt"))
    else:
        #ckpt_paths_all = list(Path(cfg.ckpt_dir).glob("*/*.ckpt"))
        ckpt_paths_all = glob.glob(os.path.join(cfg.exp_dir, '*', "*", "*.ckpt"))

    if not ckpt_paths_all:
        raise RuntimeError(f"No checkpoints found under: {cfg.exp_dir}")

    ckpt_paths_all = [Path(p) for p in ckpt_paths_all]
    folds = list(set([os.path.basename(os.path.dirname(p)) for p in ckpt_paths_all]))
    # sort in numerical order
    folds = sorted(folds, key=lambda x: int(x))
    print(f"Found folds: {folds}")

    dfs = []
    for fold in folds:
        ckpt_paths = [p for p in ckpt_paths_all if os.path.basename(os.path.dirname(p)) == fold]
        print(f"Processing fold {fold} with {len(ckpt_paths)} checkpoints.")
        # ---------- pick exactly one checkpoint using the tie-break rules ----------
        best_ckp = select_best_checkpoint(ckpt_paths)
        ckpt_name = os.path.basename(best_ckp)
        print(f"Selected checkpoint for fold {fold}: {ckpt_name}")
        ckpt_paths = [best_ckp]
        print(f"checkpoint paths: {ckpt_paths}")
        # --------------------------------------------------------------------------

        
        for ckp_path in ckpt_paths:
            # load the config that was used during training
            used_training_config_path = glob.glob(os.path.join(cfg.exp_dir, "*/config.yaml"))[0]
            print(f"Using config: {used_training_config_path}")
            used_training_cfg = OmegaConf.load(used_training_config_path)

            # --- Backcompat for older runs that didn't have weighted sampling in the YAML ---
            # Safely remove keys that may not exist
            if "trainer" in used_training_cfg:
                if "logger" in used_training_cfg.trainer:
                    used_training_cfg.trainer.pop("logger")
                if "callbacks" in used_training_cfg.trainer:
                    used_training_cfg.trainer.pop("callbacks")

            # Ensure the DataModule has the new args with safe defaults
            # (OmegaConf containers support "in" checks just like dicts.)
            if "module" in used_training_cfg.data:
                dm = used_training_cfg.data.module
                if "use_weighted_sampling" not in dm:
                    dm.use_weighted_sampling = False           # default off to match old runs
                if "weight_scaling_range" not in dm:
                    dm.weight_scaling_range = [1.0, 10.0]      # harmless default if your code expects it

            # (Optional) If you added a new head switch but old configs don't have it:
            if "classification_head_type" not in used_training_cfg.model:
                used_training_cfg.model.classification_head_type = "timm"  # keep original behavior
            # --------------------------------------------------------------------------

            # used_training_cfg.trainer.pop("logger")
            # used_training_cfg.trainer.pop("callbacks")


            used_training_cfg.model.metrics = cfg.metrics  # overwrite metrics
            # NOTE: sumin added fold setting to match the checkpoint
            used_training_cfg.data.module.fold = os.path.basename(os.path.dirname(ckp_path))

            # instantiate the model using this config
            model = instantiate(used_training_cfg.model)
            # load the weights
            model.load_state_dict(torch.load(ckp_path)["state_dict"])
            model.eval()

            # instantiate the dataset (LightningDataModule)
            datamodule = instantiate(used_training_cfg.data).module

            # >>>>>>>>>>>>>>>>> CHANGE: run predict on the VALIDATION set <<<<<<<<<<<<<<<<<
            if not hasattr(datamodule, "val_dataloader"):
                raise RuntimeError("DataModule has no val_dataloader(); cannot run on validation set.")

            # point predict to the val loader
            datamodule.predict_dataloader = datamodule.val_dataloader
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # instantiate the trainer and also pass the new metrics
            trainer = instantiate(used_training_cfg.trainer)


            # run trainer.predict on VAL
            y, y_hat = zip(*trainer.predict(model, datamodule))
            y = torch.cat(y)
            y_hat = torch.cat(y_hat)
            y_pred = torch.argmax(y_hat, dim=1)
            #===================HARDCODING FOR CONFUSION MATRIX============================#
            # ---- Map GT labels to contiguous indices defined by CLASS_NAMES ----
            # Build raw-id -> index mapping (ignore -1 if used as unlabeled)
            id2idx = {raw: i for i, raw in enumerate(RAW_TRUE_LABEL_IDS) if raw != -1}

            y_raw_np = y.cpu().numpy()
            y_idx_np = np.array([id2idx.get(int(v), -1) for v in y_raw_np], dtype=int)

            # mask out unlabeled (-1) for metrics/CM consistency
            mask = y_idx_np != -1
            if not mask.all():
                print(f"[INFO] Ignoring {(~mask).sum()} unlabeled samples (y == -1) for metrics/CM.")

            y_idx_t = torch.from_numpy(y_idx_np[mask]).to(y.device)
            y_pred_idx_t = torch.argmax(y_hat[mask], dim=1)  # predictions already 0..C-1
            # -------------------------------------------------------------------
            #===============================================================================


            # # Define classification metrics
            # acc_metric = Accuracy(task='multiclass', num_classes=used_training_cfg.model.num_classes)
            # f1_metric = F1Score(task='multiclass', num_classes=used_training_cfg.model.num_classes, average='macro')
            # auroc_metric = AUROC(task='multiclass', num_classes=used_training_cfg.model.num_classes)
            # ap_metric = AveragePrecision(task='multiclass', num_classes=used_training_cfg.model.num_classes)
            # balanced_acc_metric = BalancedAccuracy(num_classes=used_training_cfg.model.num_classes)

            # balanced_acc = balanced_acc_metric(y_pred, y)
            # acc = acc_metric(y_pred, y)
            # f1 = f1_metric(y_pred, y)
            # auroc = auroc_metric(y_hat, y)  # use logits for AUROC
            # ap = ap_metric(y_hat, y)        # use logits for Average Precision

            # print(f"Val Balanced Accuracy: {balanced_acc:.4f}")
            # print(f"Val Accuracy: {acc:.4f}")
            # print(f"Val F1 Score: {f1:.4f}")
            # print(f"Val AUROC: {auroc:.4f}")
            # print(f"Val Average Precision: {ap:.4f}")
            #===========================Confusion Matrix hardcoding edit====#
            # Define classification metrics using the mapped labels
            num_classes = len(CLASS_NAMES)
            acc_metric = Accuracy(task='multiclass', num_classes=num_classes)
            f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro')
            auroc_metric = AUROC(task='multiclass', num_classes=num_classes)
            ap_metric = AveragePrecision(task='multiclass', num_classes=num_classes)
            balanced_acc_metric = BalancedAccuracy(num_classes=num_classes)

            balanced_acc = balanced_acc_metric(y_pred_idx_t, y_idx_t)
            acc = acc_metric(y_pred_idx_t, y_idx_t)
            f1 = f1_metric(y_pred_idx_t, y_idx_t)
            auroc = auroc_metric(y_hat[mask], y_idx_t)  # logits vs mapped labels
            ap = ap_metric(y_hat[mask], y_idx_t)        # logits vs mapped labels



            #### FOR CONFIDICENCE INDERVALS 
            # ===== 95% bootstrap CIs (per fold) =====
            num_classes = len(CLASS_NAMES)
            y_true_idx_np = y_idx_np[mask]                               # (N,)
            y_pred_idx_np = y_pred_idx_t.cpu().numpy()                   # (N,)
            y_score_np    = y_hat[mask].detach().cpu().numpy()           # (N,C)

            # Point estimates (numpy-based, should match your torchmetrics above)
            metrics_point = _compute_metrics_numpy(y_true_idx_np, y_pred_idx_np, y_score_np, num_classes)

            # Bootstrap CIs (B resamples)
            cis = _bootstrap_ci(y_true_idx_np, y_pred_idx_np, y_score_np, num_classes, B=2000, seed=123)

            # Print with CIs
            for k in ["Balanced Accuracy", "Accuracy", "F1 Score", "AUROC", "Average Precision"]:
                v = metrics_point[k]
                lo, hi = cis[k]
                print(f"{k}: {v:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")

            # Save as CSV next to your existing per-fold XLSX
            metrics_ci_df = pd.DataFrame({
                "Metric": ["Balanced Accuracy", "Accuracy", "F1 Score", "AUROC", "Average Precision"],
                "Value": [metrics_point["Balanced Accuracy"], metrics_point["Accuracy"], metrics_point["F1 Score"], metrics_point["AUROC"], metrics_point["Average Precision"]],
                "CI_low": [cis["Balanced Accuracy"][0], cis["Accuracy"][0], cis["F1 Score"][0], cis["AUROC"][0], cis["Average Precision"][0]],
                "CI_high": [cis["Balanced Accuracy"][1], cis["Accuracy"][1], cis["F1 Score"][1], cis["AUROC"][1], cis["Average Precision"][1]],
            })
            metrics_out_path_ci = os.path.join(
                cfg.exp_dir, f"classification_metrics_VAL_f{fold}_{ckpt_name.split('.ckpt')[0]}_with_CI.csv"
            )
            metrics_ci_df.to_csv(metrics_out_path_ci, index=False)
            print(f"✅ Saved VAL metrics + 95% CI to {metrics_out_path_ci}")
            # ========================================

            #==============================================================####

            #============================= Confusion matrix =======================================================# 
            # … after computing y, y_hat, y_pred and metrics …
            # --- Confusion matrix assets per fold ---
            # class_names = _get_class_names(used_training_cfg.model)
            # cm_prefix = os.path.join(
            #     cfg.exp_dir,
            #     f"confusion_matrix_VAL_f{fold}_{ckpt_name.split('.ckpt')[0]}"
            # )
            # _save_confusion_matrix_assets(
            #     y_true=y.cpu().numpy(),
            #     y_pred=y_pred.cpu().numpy(),
            #     class_names=class_names,
            #     out_prefix=cm_prefix,
            #     fold_title=fold
            # )
            class_names = CLASS_NAMES  # hard-coded order
            cm_prefix = os.path.join(
                cfg.exp_dir,
                f"confusion_matrix_VAL_f{fold}_{ckpt_name.split('.ckpt')[0]}"
            )
            _save_confusion_matrix_assets(
                y_true=y_idx_np[mask],                      # mapped GT indices
                y_pred=y_pred_idx_t.cpu().numpy(),         # predicted indices
                class_names=class_names,
                out_prefix=cm_prefix,
                fold_title=fold
            )

            print(f"🧩 Saved confusion matrix CSVs and PNG to prefix: {cm_prefix}")
            #=========================================================================


            
            # Patient IDs — prefer validation dataset
            patient_ids = None
            for cand in [
                getattr(datamodule, "val_dataset", None),
                getattr(datamodule, "valid_dataset", None),
                getattr(datamodule, "dataset_val", None),
            ]:
                if cand is not None and hasattr(cand, "img_files"):
                    patient_ids = cand.img_files
                    break

            # Fallbacks (keep old behavior just in case)
            if patient_ids is None:
                try:
                    patient_ids = datamodule.test_dataset.img_files  # last resort
                except AttributeError:
                    patient_ids = getattr(datamodule, "img_files", None)

            if patient_ids is None:
                # still None -> create generic IDs
                patient_ids = [f"case_{i:05d}" for i in range(len(y_hat))]

            if len(patient_ids) != len(y_hat):
                patient_ids = patient_ids[:len(y_hat)]
            #===========(Optional) Save both raw & mapped labels in your Excel===============================
            #                       Can use old one (above one) if you want
            # df = pd.DataFrame({
            #     "PatientID": patient_ids,
            #     "GroundTruth": y.cpu().numpy(),
            #     "Prediction": y_pred.cpu().numpy(),
            #     "Logits": [log.tolist() for log in y_hat.cpu()],
            # })
            df = pd.DataFrame({
                "PatientID": np.array(patient_ids)[mask],
                "GT_raw": y_raw_np[mask],
                "GT_idx": y_idx_np[mask],
                "Pred_idx": y_pred_idx_t.cpu().numpy(),
                "Logits": [log.tolist() for log in y_hat[mask].cpu()],
            })



            #==================================================================================================
            out_path = os.path.join(cfg.exp_dir, f"classification_predictions_VAL_f{fold}_{ckpt_name}.xlsx")
            df.to_excel(out_path, index=False)
            print(f"✅ Saved VAL predictions to {out_path}")


            #CONFIDANCE INTERVAL==================
            # --- Use the numpy/CI metrics as the single source of truth for saving ---
            metrics_point = _compute_metrics_numpy(y_true_idx_np, y_pred_idx_np, y_score_np, num_classes)
            cis = _bootstrap_ci(y_true_idx_np, y_pred_idx_np, y_score_np, num_classes, B=2000, seed=123)

            # Per-fold CSV (with CI)
            metrics_ci_df = pd.DataFrame({
                "Metric": ["Balanced Accuracy", "Accuracy", "F1 Score", "AUROC", "Average Precision"],
                "Value": [metrics_point["Balanced Accuracy"], metrics_point["Accuracy"], metrics_point["F1 Score"], metrics_point["AUROC"], metrics_point["Average Precision"]],
                "CI_low": [cis["Balanced Accuracy"][0], cis["Accuracy"][0], cis["F1 Score"][0], cis["AUROC"][0], cis["Average Precision"][0]],
                "CI_high": [cis["Balanced Accuracy"][1], cis["Accuracy"][1], cis["F1 Score"][1], cis["AUROC"][1], cis["Average Precision"][1]],
                "Fold": int(fold),
                "Checkpoint": ckpt_name,
            })
            # Save per-fold metrics with CI
            metrics_out_path_ci = os.path.join(
                cfg.exp_dir, f"classification_metrics_VAL_f{fold}_{ckpt_name.split('.ckpt')[0]}_with_CI.csv"
            )
            metrics_ci_df.to_csv(metrics_out_path_ci, index=False)
            print(f"✅ Saved VAL metrics + 95% CI to {metrics_out_path_ci}")

            # Keep a copy for aggregation
            dfs.append(metrics_ci_df.copy())
            #REPLACED =
            # metrics_df = pd.DataFrame({
            #     "Metric": ["Balanced Accuracy", "Accuracy", "F1 Score", "AUROC", "Average Precision"],
            #     "Value": [balanced_acc.item(), acc.item(), f1.item(), auroc.item(), ap.item()]
            # })

            # metrics_out_path = os.path.join(cfg.exp_dir, f"classification_metrics_VAL_f{fold}_{ckpt_name}.xlsx")
            # metrics_df.to_excel(metrics_out_path, index=False)
            # print(f"✅ Saved VAL metrics to {metrics_out_path}")

            # metrics_df.rename(columns={"Value": f"Fold{fold}"}, inplace=True)
            # dfs.append(metrics_df)

            #=======================



    ### ALSO CONFIDANCE INTERVALS

    # ========= Aggregate across folds with CIs =========
    if dfs:
        # Stack per-fold rows
        per_fold = pd.concat(dfs, ignore_index=True)

        # Make a wide table: for each Metric, columns per fold value + CI
        # e.g., Fold0_Value, Fold0_CI_low, Fold0_CI_high, Fold1_..., ...
        wide = []
        for metric in per_fold["Metric"].unique():
            sub = per_fold[per_fold["Metric"] == metric].sort_values("Fold")
            row = {"Metric": metric}
            for _, r in sub.iterrows():
                f = int(r["Fold"])
                row[f"Fold{f}_Value"] = r["Value"]
                row[f"Fold{f}_CI_low"] = r["CI_low"]
                row[f"Fold{f}_CI_high"] = r["CI_high"]
            # Across-fold summary (mean + t-interval on fold means)
            vals = sub["Value"].to_numpy(dtype=float)
            k = len(vals)
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1)) if k > 1 else float("nan")
            # t-interval for small k (if k>1)
            if k > 1:
                import math
                from math import sqrt
                # t_{0.975, k-1} approx via scipy would be ideal; quick fallback ~1.96 if k>=30
                # For small k, a conservative fallback: 2.132 for k=5, 2.776 for k=3, etc.
                # We'll provide a tiny helper table for common k; else default to 1.96
                t_lut = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
                t_crit = t_lut.get(k, 1.96)
                half = t_crit * (std / sqrt(k))
                lo, hi = mean - half, mean + half
            else:
                lo, hi = float("nan"), float("nan")

            row["AcrossFolds_Mean"] = mean
            row["AcrossFolds_95CI_low"] = lo
            row["AcrossFolds_95CI_high"] = hi
            row["AcrossFolds_Nfolds"] = k
            wide.append(row)

        final_df = pd.DataFrame(wide)

        # Stable, non-misleading filename (don’t include last fold/ckpt in the name)
        dest = os.path.join(cfg.exp_dir, "VAL_metrics_all_folds_with_CI.csv")
        final_df.to_csv(dest, index=False)
        print(f"✅ Saved aggregated VAL metrics (with per-fold CIs + across-fold mean CI) to {dest}")

    ## REPLACE
    # # concat dfs where columns are folds
    # if dfs:
    #     # concat across folds
    #     if len(dfs) == 1:
    #         all_metrics_df = dfs[0]
    #     else:
    #         all_metrics_df = dfs[0]
    #         for df in dfs[1:]:
    #             all_metrics_df = all_metrics_df.merge(df, on="Metric", how="outer")

    #     dest = os.path.join(cfg.exp_dir, f"VAL_classification_metrics_VAL_f{fold}_{ckpt_name.split('.')[0]}_all.csv")
    #     all_metrics_df.to_csv(dest, index=False)
    #     print(f"✅ Saved VAL predictions (all folds) to {dest}")

if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()