#!/bin/bash

cd visualization

DATA_BASE=$1
seq_name=$(basename $DATA_BASE)
suffix='_render_scene'

echo "Path in: $DATA_BASE"
echo "Renderring sequence name: $seq_name"

nohup \
    python render_scene.py \
    --pkl_name          $seq_name \
    --base_path         $DATA_BASE \
    --img_base_path     $DATA_BASE/rgb_data/${seq_name}"_imgs" \
    --scene_pc_base_path     $DATA_BASE/lidar_data/lidar_frames_rot \
    --draw_coco17 \
    --draw_coco17_kps \
    --draw_smpl \
    --draw_human_pc \
> $DATA_BASE/rgb_data/$seq_name$suffix".log" \
2>&1 &
    # --draw_scene_pc \
