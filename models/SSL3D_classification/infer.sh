#!/bin/bash
#SBATCH --job-name=ResEnc_45kVoCo_lr1e-3_5folds
#SBATCH --time=24:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/ResEnc_45kVoCo_lr1e-3_5folds_%j.out
#SBATCH --error=logs/ResEnc_45kVoCo_lr1e-3_5folds_%j.err

# Load micromamba shell function
eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"

# Activate your environment
micromamba activate ssl3d

export EXPERIMENT_LOCATION=/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset002_AVM_T1/ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-3_5folds
export WANDB_DIR=$EXPERIMENT_LOCATION/wandb
export WANDB_CACHE_DIR=$EXPERIMENT_LOCATION/wandb_cache
export TORCH_HOME=$EXPERIMENT_LOCATION/torch_cache
export TMPDIR=/hpf/projects/jquon/tmp/tmp
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TORCH_HOME"

cd /hpf/projects/jquon/models/SSL3D_classification

exp_dir='/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset002_AVM_T1/ResEnc_no_pretrained_cls_DblWarmup_lr_1e-3/AVM_T1/checkpoints'

python inference_val_dataset.py exp_dir=$exp_dir
python inference_last_ckpt_cls.py exp_dir=$exp_dir