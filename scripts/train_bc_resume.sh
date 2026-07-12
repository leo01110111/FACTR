#!/bin/bash

CUDA_DEVICE_ID=0

task_config=ur_left
buffer_path=$(pwd)/processed_data/box-in-box/buf.pkl

# resume: curriculum already completed at step 20000 (scale reached stop_scale=0),
# so continue at sharp images (scheduler=no => scale 0) for the extension
space_config=pixel
scheduler_config=no
operator_config=blur
start_scale=5
stop_scale=0

feature_path=$(pwd)/visual_features/vit_base/SOUP_1M_DH.pth
batch_size=32
wandb_entity=leokswang-carnegie-mellon-university

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python factr/train_bc_policy.py \
exp_name=test_resume \
max_iterations=40000 \
agent.features.restore_path=$feature_path \
buffer_path=$buffer_path \
task=$task_config \
batch_size=$batch_size \
curriculum.space=$space_config \
curriculum.operator=$operator_config \
curriculum.scheduler=$scheduler_config \
curriculum.start_scale=$start_scale \
curriculum.stop_scale=$stop_scale \
wandb.debug=False \
wandb.entity=$wandb_entity
