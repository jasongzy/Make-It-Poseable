#!/usr/bin/env bash
set -e

train=(torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29501 train.py
    --num_workers 8 --remesh_type project
    --extra_char_path data/Mixamo/character_humanrig_refined)

# Latent posing
"${train[@]}" --epochs 5 --num_poses_target 5 --use_unremeshed_target 0 --expname mip/base

# Adaptive completion
"${train[@]}" --epochs 3 --num_poses_target 3 --num_latents_add 512 \
    --resume output/mip/base --resume_strict 0 --expname mip/completion
