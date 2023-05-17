################################################################################
# File: /data_processing.py                                                    #
# Created Date: Monday July 4th 2022                                           #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
# 2023-04-03	A simplified verstion for SLOPER4D dataset processing          #
################################################################################

import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath

import numpy as np
from scipy.spatial.transform import Rotation as R
import configargparse

sys.path.append(dirname(split(abspath( __file__))[0]))

from utils.tsdf_fusion_pipeline import VDBFusionPipeline
from utils import poses_to_joints, mocap_to_smpl_axis, sync_lidar_mocap
from utils import save_json_file, read_json_file, load_csv_data, print_table

field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']

def load_time(folder, index=None):
    """
    It takes a folder name and returns a list of all the times in that folder
    
    Args:
      folder: the folder where the pcd files are stored
      index: the index of the point cloud to be loaded. If None, all point clouds will be loaded.
    """
    file_list = os.listdir(folder)
    time = []
    for f in file_list:
        if f.endswith('.pcd'):
            try:
                time.append(float(f[:-4].replace('_', '.')))
            except Exception as e:
                print(f'--> {f} is not a PCD file with timestamp!')
                exit(-1)
            
    time = np.array(sorted(time))
    return time if index is None else time[index]

def traj_to_kitti_main(lidar_trajs, start, end, root_dir = None):
    """
    > It takes in a trajectory matrix, and returns a matrix of poses in the KITTI format
    
    Args:
      lidar_trajs: the trajectory data from the lidar
      start: the starting frame of the trajectory
      end: the last frame to be converted
      root_dir: the directory where the data is stored.
    
    Returns:
      the poses in the kitti format.
    """

    lidar_rots = R.from_quat(lidar_trajs[start:end, 4: 8]).as_matrix()
    lidar_trans = lidar_trajs[start:end, 1:4].reshape(-1, 3, 1)

    kitti_format_poses = np.concatenate(
        (lidar_rots, lidar_trans), axis=2).reshape(-1, 12)

    if root_dir is not None:
        poses_path = os.path.join(root_dir, 'lidar_data', 'poses.txt')
        np.savetxt(poses_path, kitti_format_poses, fmt='%.6f')
        print(f'Frame {start} to {end-1} saved in {poses_path}')

    return kitti_format_poses

def save_trajs(save_dir, rot, pos, mocap_id, comments='first', framerate=100):
    # lidar_file = os.path.join(save_dir, 'lidar_traj.txt')
    save_file = os.path.join(save_dir, f'{comments}_person_traj.txt')
    init = np.array([[-1,0,0],[0,0,1],[0,1,0]])

    t = pos[:, 0] @ init.T
    r = R.from_rotvec(rot[:, :3]).as_quat()
    time = np.array(mocap_id).reshape(-1, 1)/framerate    # -->second
    
    save_data = np.concatenate((np.array(mocap_id).reshape(-1, 1), t, r, time), axis=1)
    np.savetxt(save_file, save_data, fmt=field_fmts)

    print('Save traj in: ', save_file)

    return t 

def check_sync_valid(keytime_a, keytime_b, min_time_gap = 50, offset=1.5):
    gap_time = abs(keytime_a[-1] - keytime_a[0]) - abs(keytime_b[-1] - keytime_b[0])
    if keytime_a[-1] - keytime_a[0] > min_time_gap and abs(gap_time) < offset:
        return True, gap_time
    else:
        return False, 0

def make_sensor_params(sensor, gap_frame):
    info = f"{sensor['syncid'][gap_frame]} ({sensor['times'][sensor['syncid'][gap_frame]]}s) " if gap_frame>0 else gap_frame
    return {"Framerate"       : sensor['framerate'],
            "Total frames"    : len(sensor['times']),
            "Start frame"     : f"{sensor['syncid'][0]} ({sensor['times'][sensor['syncid'][0]]:.3f}s)",
            "- Keyframe"      : f"{sensor['keyid'][0]} ({sensor['keytime'][0]:.3f}s)",
            "- Gapframe"      : info,
            "- Keyframe2"     : f"{sensor['keyid'][-1]} ({sensor['keytime'][-1]:.3f}s)",
            "End frame"       : f"{sensor['syncid'][-1]} ({sensor['times'][sensor['syncid'][-1]]:.3f}s)",
            "Relative keytime": f"{sensor['keytime'][-1] - sensor['keytime'][0]:.3f}"}

def update_data(save_data, person, save_trans, save_pose):
    betas = np.array([0.0] * 10)
    if 'betas' in dataset_params[f'{person}_person']:
        betas = dataset_params[f'{person}_person']['betas']
    if 'gender' in dataset_params[f'{person}_person']:
        gender = dataset_params[f'{person}_person']['gender']
    else:
        gender = 'male'
    if f'{person}_person' not in save_data:
        save_data[f'{person}_person'] = {}

    save_data[f'{person}_person'].update({
                                'beta': betas, 
                                'gender': gender, 
                                'pose': save_pose, 
                                'mocap_trans': save_trans})

def save_sync_data(root_folder, start=0, end=np.inf, gap_frame=-1):
    """
    > It takes in the path to the data folder, and returns the poses of the two people, the translations
    of the two people, and the frame numbers of the lidar data
    
    Args:
      root_folder: the path to the folder containing the data
      start: the start frame of the video.
      end: the last frame to be processed. Defaults to inf
    """
    first_data  = True
    second_data = True

    lidar_dir   = glob(root_folder + '/lidar_data/*lidar_frames_rot')[0]
    json_path   = glob(root_folder + '/*dataset_params.json')[0]

    # lidar_trajectory = glob(root_folder + '/*lidar_trajectory.txt')[0]
    save_dir = os.path.join(root_folder, 'synced_data')
    param_file = os.path.join(save_dir, 'humans_param.pkl')
    os.makedirs(save_dir, exist_ok=True)

    dataset_params = read_json_file(json_path)
    lidar = {}
    mocap = {}

    lidar['times'] = load_time(lidar_dir)
    lidar['framerate'] = 1 / np.diff(lidar['times']).mean()
    mocap['framerate'] = dataset_params['mocap_framerate']

    dataset_params['lidar_framerate'] = lidar['framerate']
    save_json_file(json_path, dataset_params)

    try:
        if os.path.exists(param_file):
            with open(param_file, "rb") as f:
                save_data = pickle.load(f)
            print(f"Synced data loaded in: {param_file}")
        else:
            save_data = {}
    except:
        save_data = {}

    def load_csv(search_str='*'):
        rot_csv = glob(root_folder + f'/mocap_data/*{search_str}_rot*.csv')[0]
        pos_csv = glob(root_folder + f'/mocap_data/*{search_str}_*pos.csv')[0]
        pos, rot, col = load_csv_data(rot_csv, pos_csv)
        mocap['times'] = rot[:, 0]
        return pos, rot, col
    
    try:
        first_pos, first_rot, col_names = load_csv('first')
    except BaseException:
        print('=======================')
        print('No first person data!!')
        print('=======================')
        first_data = False
        
    try:
        second_pos, second_rot, col_names = load_csv('second')
    except:
        try:
            second_pos, second_rot, col_names = load_csv('*')
        except:
            print('=======================')
            print('No second person data!!')
            print('=======================')
            second_data = False

    if not first_data and not second_data:
        print("No mocap data in './mocap_data/'!!!")
        exit(0)
    
    lidar['keytime']  = dataset_params['lidar_sync']   # lidar keytime for synchronization
    mocap['keytime']  = dataset_params['mocap_sync']   # mocap keytime for synchronization

    def get_keyid(sensor: dict):
        keyid = []
        for kt in sensor['keytime']:
            keyid.append(np.where(abs(sensor['times'] - kt) < 1/sensor['framerate'])[0][0])
        return keyid
    
    lidar['keyid'] =  get_keyid(lidar)
    mocap['keyid'] =  get_keyid(mocap)

    double_sync, gap_time = check_sync_valid(lidar['keytime'], mocap['keytime'])
    
    # 1. sync. data
    lidar['syncid'], mocap['syncid'] = sync_lidar_mocap(lidar['times'], mocap['times'], 
                                                  lidar['keytime'][0], mocap['keytime'][0], 
                                                  1/mocap['framerate'])

    # 2 choose start time
    if 'start' in dataset_params:
        start =  dataset_params['start']
    elif start == 0:
        start = int(lidar['keyid'][0]) - round(lidar['framerate']) * 2
    else:
        start = min(int(lidar['keyid'][0]) - round(lidar['framerate']) * 2, start) 

    start = lidar['syncid'].index(start) if start in lidar['syncid'] else 0
    end   = lidar['syncid'].index(end) if end in lidar['syncid'] else len(lidar['syncid']) - 1

    # 3. if frame dropping occurs in mocap data
    gap_frame = dataset_params['gap_frame'] if 'gap_frame' in dataset_params else gap_frame
    if double_sync and gap_frame > 0:
        lidar['syncid2'], mocap['syncid2'] = sync_lidar_mocap(lidar['times'], mocap['times'], 
                                                        lidar['keytime'][-1], mocap['keytime'][-1],
                                                        1/mocap['framerate'])
        offset_frame = lidar['syncid2'][0] - lidar['syncid'][0]
        # dropping frames in mocap data
        if offset_frame > 0:
            if lidar['syncid'][end] in lidar['syncid2']:
                from_end = lidar['syncid2'].index(lidar['syncid'][end]) 
            else:
                from_end = len(lidar['syncid']) - 1
            
            lidar['syncid'] = lidar['syncid'][start: start + gap_frame] + \
                           lidar['syncid2'][start + gap_frame - offset_frame: from_end + 1]
            mocap['syncid'] = mocap['syncid'][start: start + gap_frame] + \
                           mocap['syncid2'][start + gap_frame - offset_frame: from_end + 1]
            assert (np.diff(np.array(lidar['syncid'])) != 1).sum() == 0, "LiDAR data is not "
    else:
        lidar['syncid'] = lidar['syncid'][start:end+1]
        mocap['syncid'] = mocap['syncid'][start:end+1]

    # 4. print the necessary information
    dataset_params['lidar'] = make_sensor_params(lidar, gap_frame)
    dataset_params['mocap'] = make_sensor_params(mocap, gap_frame)
    save_json_file(json_path, dataset_params)
    print_table("Synchronization Parameters", 
                ["Parameter", "LiDAR", "Mocap(IMU)"],
                [dataset_params['lidar'], dataset_params['mocap']])
    print(f'Reletive keytime offset: {-gap_time:.3f}\n')

    if lidar['syncid'][0] > 1000:
        print("Note that the starting frame for the LiDAR is > 1000.")
        print("It is recommended to manually set the starting frame number.")

    # start to sync all the data
    if first_data:
        sync_pose, _  = mocap_to_smpl_axis(first_rot[mocap['syncid']], 
                                            fix_orit = False, 
                                            col_name=col_names)
        # 2. save synced mocap trans
        trans = save_trajs(save_dir, sync_pose, first_pos[mocap['syncid']], mocap['syncid'], mocap['framerate'])

        joints, verts  = poses_to_joints(sync_pose[:1], return_verts=True) 
        feet_center    = (joints[0, 7] + joints[0, 8])/2
        feet_center[2] = verts[..., 2].min()
        trans      -= trans[0] + feet_center  # 使得第一帧位于原点

        # 3. save synced data
        update_data(save_data, 'first', trans, sync_pose)

    if second_data:
        fix_orit = True if not first_data else False

        sync_pose_2nd, delta_r = mocap_to_smpl_axis(second_rot[mocap['syncid']], 
                                           fix_orit=fix_orit,
                                           col_name=col_names)

        second_trans = save_trajs(save_dir, sync_pose_2nd, 
                                  second_pos[mocap['syncid']] @ delta_r.T, 
                                  mocap['syncid'], 'second', mocap['framerate'])
        update_data(save_data, 'second', second_trans, sync_pose_2nd)
        if not first_data:
            trans = second_trans
            sync_pose = sync_pose_2nd

    save_data['frame_num'] = lidar['syncid']
    save_data['framerate'] = lidar['framerate']

    with open(param_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {param_file}")
    
    return sync_pose, trans, lidar['syncid'], param_file   # type: ignore

def add_lidar_traj(synced_data_file, lidar_traj):
    """
    It takes a synced data file and a lidar trajectory, and saves the lidar trajectory in the synced
    data file
    
    Args:
      synced_data_file: the file that contains the synced data
      lidar_traj: a list of lidar poses, each of which is a list of [ID, x, y, z, qx, qy, qz, qw, timestamp]
    """

    with open(synced_data_file, "rb") as f:
        save_data = pickle.load(f)

    save_data['first_person'] = {'lidar_traj': lidar_traj}

    with open(synced_data_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"Lidar traj saved in {synced_data_file}")


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, help="The data's root directory")

    parser.add_argument("--traj_file", type=str, default='lidar_trajectory.txt')

    parser.add_argument("--params_file", type=str, default='dataset_params.json')

    parser.add_argument('-S', "--start_idx", type=int, default=0,
                        help='The start frame index in LiDAR for processing, specified when sychronization time is too late')

    parser.add_argument('-E', "--end_idx", type=int, default=np.inf,
                        help='The end frame index in LiDAR for processing, specified when sychronization time is too early.')

    parser.add_argument("-VS", "--voxel_size", type=float, default=0.06, 
                        help="The voxel filter parameter for TSDF fusion")

    parser.add_argument("--skip_frame", type=int, default=8, 
                        help='The everay n frame used for mapping')
    
    parser.add_argument('--tsdf', action='store_true',
                        help="Use VDB fusion to build the scene mesh")
    
    parser.add_argument('--sdf_trunc', type=float, default=0.25,
                        help="The trunction distance for SDF funtion")

    parser.add_argument('--sync', action='store_true', 
                        help='Synced all data and save a pkl based on the params_file')

    args, opts    = parser.parse_known_args()
    root_folder   = args.root_folder
    traj_file     = os.path.join(root_folder, 'lidar_data', args.traj_file)
    params_file   = os.path.join(root_folder, args.params_file)
    lidar_dir     = os.path.join(root_folder, 'lidar_data', 'lidar_frames_rot')

    if os.path.exists(params_file):
        dataset_params = read_json_file(params_file)
        if 'start' in dataset_params:
            args.start_idx = dataset_params['start']
        if 'end' in dataset_params:
            args.end_idx   = dataset_params['end']

    # --------------------------------------------
    # 1. Load the LiDAR trajectory.
    # --------------------------------------------
    lidar_traj =  np.loadtxt(traj_file, dtype=float)

    args.end_idx = min(args.end_idx, lidar_traj.shape[0])

    # --------------------------------------------
    # 2. Synchronize the LiDAR and mocap data.
    # --------------------------------------------
    if args.sync:
        _, _, lidar_frameid, synced_data_file = save_sync_data(
            root_folder, 
            start = args.start_idx, 
            end   = args.end_idx)
        add_lidar_traj(synced_data_file, lidar_traj[lidar_frameid])
        mapping_start, mapping_end = lidar_frameid[0] + 100, lidar_frameid[-1] - 100
    else:
        mapping_start, mapping_end = args.start_idx, args.end_idx
        
    # --------------------------------------------
    # 3. Use TSDF fusion to build the scene mesh
    # --------------------------------------------
    if args.tsdf:
        kitti_poses = traj_to_kitti_main(lidar_traj, args.start_idx, args.end_idx)
        vdbf = VDBFusionPipeline(lidar_dir, 
                                traj_to_kitti_main(lidar_traj, mapping_start, mapping_end), 
                                start         = mapping_start,
                                end           = mapping_end,
                                map_name      = os.path.basename(root_folder),
                                sdf_trunc     = args.sdf_trunc,
                                voxel_size    = args.voxel_size, 
                                space_carving = True)
        vdbf.run(skip=args.skip_frame)
        try:
            vdbf.segment_ground()
        except Exception as e:
            print(e.args[0])

# python process_raw_data.py --root_folder <root folder> [--tsdf] [--sync]
# --tsdf,         building the scene mesh
# --sync,         synchronize lidar and imu
# --voxel_size,   The voxel filter parameter for TSDF fusion
