import os
import sys
import json

import cv2

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import RotationSpline
from scipy.spatial.transform import Rotation as R

sys.path.append('.')
sys.path.append('..')
from src import SLOPER4D_Loader


def interpolate_transform_matrices(transform_matrices, timestamps, target_ts):
    """
    Interpolates the rotation and translation components of transform matrices at a given timestamp.

    Args:
        transform_matrices (list): List of 4x4 transform matrices.
        timestamps (numpy.ndarray): List of corresponding timestamps.
        target_ts (numpy.ndarray): Target timestamp for interpolation.

    Returns:
        numpy.ndarray: Interpolated 4x4 transform matrix.

    Raises:
        ValueError: If the number of matrices is not equal to the number of timestamps.
    """
    assert len(transform_matrices) == len(timestamps) == len(target_ts), "Number of matrices should be equal to the number of timestamps."

    rotations = R.from_matrix(transform_matrices[:, :3, :3])
    translations = transform_matrices[:, :3, 3]

    rotation_spline = RotationSpline(timestamps, rotations)
    translation_spline = CubicSpline(timestamps, translations, axis=0)

    interpolated_rotation = rotation_spline(target_ts)
    interpolated_translation = translation_spline(target_ts)

    # Construct interpolated matrix
    interpolated_matrix = np.array([np.eye(4)] * len(rotations))
    interpolated_matrix[:, :3, :3] = interpolated_rotation.as_matrix()
    interpolated_matrix[:, :3, 3] = interpolated_translation

    return interpolated_matrix

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root_folder', type=str, default=None, 
        help='path to dataset')
    parser.add_argument('-I', '--interpolate', action='store_true',
        help='whether to interpolate the rotation by SLERP')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    scene_path       = args.root_folder
    scene_name       = os.path.basename(args.root_folder)
    pkl_path         = os.path.join(scene_path, scene_name+'_labels.pkl')
    data_params_path = os.path.join(scene_path, 'dataset_params.json')
    vdo_path         = os.path.join(scene_path, 'rgb_data', scene_name+'.MP4')
    
    # open video
    vid = cv2.VideoCapture()
    vid.open(vdo_path)
    assert vid.isOpened(), "Error: video not opened!"

    vid_fps = float(vid.get(cv2.CAP_PROP_FPS))
    rgb_count  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))+1
    
    # open pkl and json files
    with open(data_params_path, 'r') as f:
        data_params_data = json.load(f)
    sequence = SLOPER4D_Loader(pkl_path, return_torch=False)

    # get lidar timestamps
    time_rgb2lidar = data_params_data['lidar_sync'][0] - data_params_data['rgb_sync'][0]
    lidar_tstamp = np.array(sequence.lidar_tstamps)
    rgb_frametime = 1/vid_fps

    lidar_idx     = 0
    file_basename = []
    time_rgb_in_lidar = []
    # assign the closest timestamp the images
    
    for i in range(0, rgb_count):
        if lidar_idx >= lidar_tstamp.shape[0]: 
            break
        time_rgb = i*rgb_frametime + time_rgb2lidar

        if abs(lidar_tstamp[lidar_idx]-time_rgb) <= rgb_frametime / 2:
            time_rgb_in_lidar.append(time_rgb)
            fname = f"{i*rgb_frametime:.06f}.jpg"
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

    lenght = len(file_basename)
    sequence.data['total_frames'] = lenght

    sequence.data['RGB_info'] = {
        "width"     : data_params_data['RGB_info']['width'],
        "height"    : data_params_data['RGB_info']['height'],
        "fps"       : vid_fps,
        "intrinsics": data_params_data['RGB_info']['intrinsics'],
        "lidar2cam" : data_params_data['RGB_info']['lidar2cam'],
        "dist"      : data_params_data['RGB_info']['dist']
    }

    # Use Slerp to calculate the world2lidar transformation time_rgb_in_lidar
    lidar_traj   = sequence.data['first_person']['lidar_traj'].copy()[:lenght]
    lidar_tstamp = lidar_tstamp[:lenght]

    lidar2world  = np.array([np.eye(4)] * lenght)
    lidar2world[:, :3, :3] = R.from_quat(lidar_traj[:, 4: 8]).as_matrix()
    lidar2world[:, :3, 3:] = lidar_traj[:, 1:4].reshape(-1, 3, 1)

    if args.interpolate:
        print('Rotations will be interpolated...')
        offset = lidar_tstamp[0] - time_rgb_in_lidar[0]
        lidar2world = interpolate_transform_matrices(lidar2world, lidar_tstamp, time_rgb_in_lidar + offset)
    world2lidar    = np.array([np.eye(4)] * lenght)
    world2lidar[:, :3, :3] = R.from_matrix(lidar2world[:, :3, :3]).inv().as_matrix()
    world2lidar[:, :3, 3:] = -world2lidar[:, :3, :3] @ lidar2world[:, :3, 3:].reshape(-1, 3, 1)

    # update the rgb frames
    rf = sequence.get_rgb_frames()
    rf['file_basename'] = file_basename
    rf['lidar_tstamps'] = rf['lidar_tstamps'][:lenght]

    rf['bbox']     = rf['bbox'][:lenght]
    rf['skel_2d']  = rf['skel_2d'][:lenght]
    # rf['cam_pose'] = rf['cam_pose'][:lenght]
    rf['cam_pose'] = data_params_data['RGB_info']['lidar2cam'] @ world2lidar

    sequence.save_pkl(overwrite=True)
        
    print("max offset between lidar and RGB: ", (lidar_tstamp - np.array(time_rgb_in_lidar)).max())
        