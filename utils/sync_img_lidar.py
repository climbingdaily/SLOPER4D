import os
import sys
import json
import pickle

import cv2

sys.path.append('.')
sys.path.append('..')
from src import SLOPER4D_Loader

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # basics
    parser.add_argument('--base_path', default= '/wd8t/sloper4d_publish',
        type=str, #required=True,
        help='path to dataset'
    )
    parser.add_argument("--scene_name", default='seq003_street_002',
        type=str, #required=True 
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    scene_path       = os.path.join(args.base_path, args.scene_name)
    pkl_path         = os.path.join(scene_path, args.scene_name+'_labels.pkl')
    data_params_path = os.path.join(scene_path, 'dataset_params.json')
    vdo_path         = os.path.join(scene_path, 'rgb_data', args.scene_name+'.MP4')
    
    # open video
    vid = cv2.VideoCapture()
    vid.open(vdo_path)
    assert vid.isOpened(), "Error: video not opened!"

    vid_fps = float(vid.get(cv2.CAP_PROP_FPS))
    endIdx  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    
    # open pkl and json files
    with open(data_params_path, 'r') as f:
        data_params_data = json.load(f)
    sequence = SLOPER4D_Loader(pkl_path, return_torch=False)

    # get lidar timestamps
    time_rgb2lidar = data_params_data['lidar_sync'][0] - data_params_data['rgb_sync'][0]
    lidar_tstamp = sequence.lidar_tstamps

    time_dev = 1 / vid_fps / 2

    lidar_idx     = 0
    file_basename = []
    img_devs      = []

    # assign the closest timestamp the images
    for i in range(0, endIdx):
        time_rgb = i/vid_fps + time_rgb2lidar
        if lidar_idx >= lidar_tstamp.shape[0]: 
            break
        # while(lidar_tstamp[lidar_idx] < time_rgb and lidar_tstamp[lidar_idx] - time_rgb > time_dev): lidar_idx += 1
        if abs(lidar_tstamp[lidar_idx]-time_rgb) > time_dev:
            continue
        img_devs.append(abs(lidar_tstamp[lidar_idx]-time_rgb))
        fname = f"{i/vid_fps:.06f}.jpg"
        file_basename.append(fname)
        lidar_idx += 1

    print("num of RGB frames:", len(file_basename))
    print("num of LiDAR frames:", lidar_tstamp.shape[0])

    if len(file_basename)!=lidar_tstamp.shape[0]:
        if lidar_idx < lidar_tstamp.shape[0]: 
            print("RGB end before final lidar frame.")
        else: 
            print("!!!!!ERROR in time alignment!!!!!")
            sys.exit(1)

    sequence.data['total_frames'] = len(file_basename)
    rf = sequence.get_rgb_frames()
    rf['file_basename'] = file_basename
    rf['lidar_tstamps'] = rf['lidar_tstamps'][:len(file_basename)]

    rf['bbox']          = rf['bbox'][:len(file_basename)]
    rf['skel_2d']       = rf['skel_2d'][:len(file_basename)]
    rf['cam_pose']      = rf['cam_pose'][:len(file_basename)]
    
    sequence.data['RGB_info'] = {
        "width"     : data_params_data['RGB_info']['width'],
        "height"    : data_params_data['RGB_info']['height'],
        "fps"       : vid_fps,
        "intrinsics": data_params_data['RGB_info']['intrinsics'],
        "lidar2cam" : data_params_data['RGB_info']['lidar2cam'],
        "dist"      : data_params_data['RGB_info']['dist']
    }

    sequence.save_pkl(overwrite=True)
        
    print("max offset between lidar and RGB: ", max(img_devs))
        