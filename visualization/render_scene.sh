#!/bin/bash

INDEX=-1
DRAW_OPTION="--draw_coco17 --draw_smpl --draw_human_pc --draw_scene_pc --draw_mask"

DATA_BASE=$1
INDEX=${2:-$INDEX}
DRAW_OPTION=${3:-$DRAW_OPTION}

seq_name=$(basename $DATA_BASE)
suffix='_render_scene'
log_path=$DATA_BASE/rgb_data/$seq_name$suffix".log"

echo "Renderring sequence in: $DATA_BASE"
echo "Print information in: $log_path"
echo "DRAW OPTION: $DRAW_OPTION"

python ./visualization/render_scene.py \
    --base_path $DATA_BASE \
    --index $INDEX $DRAW_OPTION > $log_path
