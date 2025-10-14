#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
# -------------------------------
# Ready-to-run RSNA Fold 0 Training
# -------------------------------

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssl3d

# Set PYTHONPATH so Hydra finds models
export PYTHONPATH=$(pwd):$PYTHONPATH

# -------------------------------
# Experiment paths
# -------------------------------
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export nnssl_raw="/home/jma/Documents/rsna/nnssl_dataset/nnssl_raw"
export nnssl_preprocessed="/home/jma/Documents/rsna/nnssl_dataset/nnssl_preprocessed"
export nnssl_results="/home/jma/Documents/rsna/nnssl_dataset/nnssl_results"
export EXPERIMENT_LOCATION=/home/jma/Documents/rsna/nnssl_dataset/nnssl_results/Dataset007_RSNA/RSNA_scratch_ResEnc_cls_DblWarmup_lr_1e-4_5folds_bsize8
export WANDB_DIR=$EXPERIMENT_LOCATION/wandb
export WANDB_CACHE_DIR=$EXPERIMENT_LOCATION/wandb_cache
export TORCH_HOME=$EXPERIMENT_LOCATION/torch_cache
export TMPDIR=/home/jma/Documents/rsna/tmp/tmp

# Create directories if they don't exist
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TORCH_HOME" "$TMPDIR"

# -------------------------------
# Go to repo root
# -------------------------------
cd /home/jma/Documents/rsna/models/SSL3D_classification

# -------------------------------
# Run fold 0 training
# -------------------------------
python main_5folds.py env=cluster model=resenc data=RSNA_data \
  trainer.devices=1 model.pretrained=False \
  data.module.fold=0 \
  trainer.logger.name=Fold0_RSNA_scratch_ResEnc_cls_DblWarmup_lr_1e-4_5folds_bsize2
