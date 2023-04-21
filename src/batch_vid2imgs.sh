#!/bin/bash

defalt_path=/wd8t/sloper4d_publish

dataset_root=${1:-$defalt_path}

seq_list=(
    "seq001_campus_001"
    "seq002_football_001"
    "seq003_street_002"
    "seq004_library_001"
    "seq005_library_002"
    "seq006_library_003"
    "seq007_garden_001"
    "seq008_running_001"
    "seq009_running_002"
    "seq010_park_001"
    "seq011_park_002"
    "seq012_musicsquare_001"
    "seq013_sunlightrock_001"
    "seq014_garden_002"
    "seq015_mayuehansquare_001"
)

for seq_name in "${seq_list[@]}"; do
    root_folder=$dataset_root/$seq_name
    if [ -d "$root_folder" ]; then
        echo "Processing sequence: $seq_name"
        python src/vid2imgs.py $root_folder
    else
        echo "Sequence folder not found: $root_folder"
    fi
done