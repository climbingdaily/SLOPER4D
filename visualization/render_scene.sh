#!/bin/bash

INDEX=-1

DATA_BASE=$1
INDEX=${2:-$INDEX}

seq_name=$(basename $DATA_BASE)
suffix='_render_scene'

echo "Renderring sequence in: $DATA_BASE"

python ./visualization/render_scene.py --base_path $DATA_BASE --index $INDEX \
    --draw_coco17 \
    --draw_coco17_kps \
    --draw_smpl \
    --draw_human_pc \
    --draw_mask \
    --draw_scene_pc \
> $DATA_BASE/rgb_data/$seq_name$suffix".log"
