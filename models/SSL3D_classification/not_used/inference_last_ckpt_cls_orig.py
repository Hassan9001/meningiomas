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

# -------------------- NEW: checkpoint parsing & selection --------------------
_ckpt_rx = re.compile(
    r"epoch(?P<epoch>\d+).*?"
    r"Val_loss=(?P<val_loss>\d+(?:\.\d+)?).*?"
    r"Val_(?:bal|bala|balanced)_acc=(?P<bal_acc>\d+(?:\.\d+)?)\.ckpt",
    re.IGNORECASE
)

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


@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer_cls")
def inference(cfg):
    # delete automatically created hydra logger
    try:
        Path("./main.log").unlink()
    except:
        pass

    # collect candidate checkpoints
    if cfg.fold:
        ckpt_paths = list(Path(Path(cfg.ckpt_dir) / str(cfg.fold)).glob("*.ckpt"))
    else:
        ckpt_paths = list(Path(cfg.ckpt_dir).glob("*/*.ckpt"))

    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found under: {cfg.ckpt_dir}")

    # ---------- NEW: pick exactly one checkpoint using the tie-break rules ----------
    best_ckp = select_best_checkpoint(ckpt_paths)
    ckpt_paths = [best_ckp]
    print(f"checkpoint paths: {ckpt_paths}")
    # -------------------------------------------------------------------------------

    for ckp_path in ckpt_paths:
        # load the config that was used during training
        used_training_config_path = glob.glob(os.path.join(cfg.exp_dir, "*/config.yaml"))[0]
        print(f"Using config: {used_training_config_path}")
        used_training_cfg = OmegaConf.load(used_training_config_path)
        used_training_cfg.trainer.pop("logger")
        used_training_cfg.trainer.pop("callbacks")
        used_training_cfg.model.metrics = cfg.metrics  # overwrite metrics

        # instantiate the model using this config
        model = instantiate(used_training_cfg.model)
        # load the weights
        model.load_state_dict(torch.load(ckp_path)["state_dict"])
        model.eval()

        # instantiate the dataset from the config
        dataset = instantiate(used_training_cfg.data).module
        # instantiate the trainer and also pass the new metrics
        trainer = instantiate(used_training_cfg.trainer)

        # run trainer.predict
        y, y_hat = zip(*trainer.predict(model, dataset))
        y = torch.cat(y)
        y_hat = torch.cat(y_hat)

        y_pred = torch.argmax(y_hat, dim=1)

        # Define classification metrics
        acc_metric = Accuracy(task='multiclass', num_classes=used_training_cfg.model.num_classes)
        f1_metric = F1Score(task='multiclass', num_classes=used_training_cfg.model.num_classes, average='macro')
        auroc_metric = AUROC(task='multiclass', num_classes=used_training_cfg.model.num_classes)
        ap_metric = AveragePrecision(task='multiclass', num_classes=used_training_cfg.model.num_classes)
        balanced_acc_metric = BalancedAccuracy(num_classes=used_training_cfg.model.num_classes)

        balanced_acc = balanced_acc_metric(y_pred, y)
        acc = acc_metric(y_pred, y)
        f1 = f1_metric(y_pred, y)
        auroc = auroc_metric(y_hat, y)  # use logits for AUROC
        ap = ap_metric(y_hat, y)        # use logits for Average Precision

        print(f"Test Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test Average Precision: {ap:.4f}")

        # Patient IDs
        try:
            patient_ids = dataset.test_dataset.img_files  # for datamodule-style
        except AttributeError:
            patient_ids = dataset.img_files  # direct dataset

        if len(patient_ids) != len(y_hat):
            patient_ids = patient_ids[:len(y_hat)]

        df = pd.DataFrame({
            "PatientID": patient_ids,
            "GroundTruth": y.cpu().numpy(),
            "Prediction": y_pred.cpu().numpy(),
            "Logits": [log.tolist() for log in y_hat.cpu()],
        })

        out_path = os.path.join(cfg.exp_dir, f"test_classification_predictions.xlsx")
        df.to_excel(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")

        metrics_df = pd.DataFrame({
            "Metric": ["Balanced Accuracy", "Accuracy", "F1 Score", "AUROC", "Average Precision"],
            "Value": [balanced_acc.item(), acc.item(), f1.item(), auroc.item(), ap.item()]
        })

        metrics_out_path = os.path.join(cfg.exp_dir, f"test_classification_metrics.xlsx")
        metrics_df.to_excel(metrics_out_path, index=False)
        print(f"✅ Saved metrics to {metrics_out_path}")


if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
