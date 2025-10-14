from huggingface_hub import snapshot_download
import os

#snapshot_download(repo_id="AnonRes/ResEncL-OpenMind-SwinUNETR", local_dir="./pretrained_checkpoints/SwinUNETR")
#snapshot_download(repo_id="AnonRes/ResEncL-OpenMind-SimCLR", local_dir="./pretrained_checkpoints/SimCLR")
#snapshot_download(repo_id="AnonRes/ResEncL-OpenMind-VoCo", local_dir="./pretrained_checkpoints/VoCo")
snapshot_download(repo_id="AnonRes/ResEncL-OpenMind-S3D", local_dir="./pretrained_checkpoints/S3D")