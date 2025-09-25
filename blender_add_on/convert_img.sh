#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_directory> [frame_rate]"
    exit 1
fi

input_dir=$1
output_dir="Videos"
start_number=$(ls $input_dir/*.png | sort | head -n 1 | sed 's/.*\/\([0-9]*\)\.png/\1/')


frame_rate=${2:-20}

if [ -z "$start_number" ]; then
    echo "Unable to determine start number from PNG files in $input_dir."
    exit 1
fi

output_file="${output_dir}/${input_dir##*/}.mp4"

mkdir -p $output_dir

ffmpeg -framerate $frame_rate -start_number $start_number -i "${input_dir}/%04d.png" -c:v libx264 -r 20 "$output_file"

echo "Video saved to $output_file"

