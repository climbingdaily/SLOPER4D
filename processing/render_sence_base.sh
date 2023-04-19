#!/bin/bash

# args
PROJ_PATH="/home/lyt/github/SLOPER4D/"
DATA_PATH="/wd8t/sloper4d_publish/"

DATA_BASE=$DATA_PATH"$1/"
scene=$1
suffix='_render_sence'

# environment
cd $PROJ_PATH/processing/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate detectron 

nohup \
    python render_sence.py \
    --pkl_name          $scene \
    --base_path         $DATA_BASE \
    --img_base_path     $DATA_BASE/rgb_data/$1"_imgs" \
    --scene_pc_base_path     $DATA_BASE/lidar_data/lidar_frames_rot \
    --draw_coco17 \
    --draw_coco17_kps \
    --draw_smpl \
    --draw_human_pc \
    --draw_scene_pc \
> $DATA_BASE/rgb_data/$scene$suffix".log" \
2>&1 &
