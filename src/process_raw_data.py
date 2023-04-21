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
from utils import poses_to_joints, mocap_to_smpl_axis, save_json_file, read_json_file, sync_lidar_mocap, load_csv_data

field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']

def print_table(title, headers, rows):
    # 计算每列的最大宽度
    col_widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(item)))
    
    # 打印表格标题
    title_width = sum(col_widths) + len(headers) * 3 - 1
    print(f"\n{title.center(title_width)}")
    
    # 打印表格上部边框
    print("┌" + "─" * (col_widths[0] + 2) + "┬" + "─" * (col_widths[1] + 2) + "┬" + "─" * (col_widths[2] + 2) + "┐")
    
    # 打印表头
    header_str = "│ " + " │ ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers)) + " │"
    print(header_str)
    
    # 打印表格中部分隔线
    print("├" + "─" * (col_widths[0] + 2) + "┼" + "─" * (col_widths[1] + 2) + "┼" + "─" * (col_widths[2] + 2) + "┤")
    
    # 打印表格内容
    for row in rows:
        row_str = "│ " + " │ ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " │"
        print(row_str)
        
    # 打印表格下部边框
    print("└" + "─" * (col_widths[0] + 2) + "┴" + "─" * (col_widths[1] + 2) + "┴" + "─" * (col_widths[2] + 2) + "┘")

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

def save_trajs(save_dir, rot, pos, mocap_id, comments='first'):
    """
    It takes in the rotation and position data from the mocap system, and saves it in a format that the
    [KITTI evaluation toolkit](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) can read. 

    Args:
      save_dir: the directory to save the trajectory file
      rot: rotation of the person in the world frame
      pos: the position of the person in the world frame
      mocap_id: the frame number of the mocap data
      comments: a string that will be added to the filename of the saved trajectory. Defaults to first
    
    Returns:
      The translation of the person's position.
    """
    # lidar_file = os.path.join(save_dir, 'lidar_traj.txt')
    save_file = os.path.join(save_dir, f'{comments}_person_traj.txt')

    init = np.array([[-1,0,0],[0,0,1],[0,1,0]])

    t = pos[:, 0] @ init.T
    r = R.from_rotvec(rot[:, :3]).as_quat()
    time = np.array(mocap_id).reshape(-1, 1)/100    # -->second
    
    save_data = np.concatenate((time * 100, t, r, time), axis=1)
    np.savetxt(save_file, save_data, fmt=field_fmts)

    print('Save traj in: ', save_file)

    return t 

def save_sync_data(root_folder, start=0, end=np.inf):
    """
    > It takes in the path to the data folder, and returns the poses of the two people, the translations
    of the two people, and the frame numbers of the lidar data
    
    Args:
      root_folder: the path to the folder containing the data
      start: the start frame of the video.
      end: the last frame to be processed. Defaults to inf
    """
    first_data = True
    second_data = True
    lidar_dir        = glob(root_folder + '/lidar_data/*lidar_frames_rot')[0]
    json_path        = glob(root_folder + '/*dataset_params.json')[0]

    # lidar_trajectory = glob(root_folder + '/*lidar_trajectory.txt')[0]
    save_dir = os.path.join(root_folder, 'synced_data')
    param_file = os.path.join(save_dir, 'humans_param.pkl')
    os.makedirs(save_dir, exist_ok=True)

    lidar_time = load_time(lidar_dir)
    lidar_framerate = round(1 / np.diff(lidar_time).mean())
    
    dataset_params = read_json_file(json_path)
    dataset_params['lidar_framerate'] = lidar_framerate
    save_json_file(json_path, dataset_params)

    try:
        if os.path.exists(param_file):
            with open(param_file, "rb") as f:
                save_data = pickle.load(f)
            print(f"Synced data loaded in: {param_file}")
        else:
            save_data = {}
    except Exception as e:
        save_data = {}

    try:
        first_rot_csv    = glob(root_folder + '/mocap_data/*first_rot*.csv')[0]
        first_pos_csv    = glob(root_folder + '/mocap_data/*first_*pos.csv')[0]
        first_pos, first_rot, col_names = load_csv_data(first_rot_csv, first_pos_csv)
        mocap_time = first_rot[:, 0]
    except BaseException:
        print('=======================')
        print('No first person data!!')
        print('=======================')
        first_data = False
        
    try:
        second_rot_csv   = glob(root_folder + '/mocap_data/*second_rot*.csv')[0]
        second_pos_csv   = glob(root_folder + '/mocap_data/*second_*pos.csv')[0]
        second_pos, second_rot, col_names = load_csv_data(second_rot_csv, second_pos_csv)
        mocap_time = second_rot[:, 0]
    except Exception:
        try:
            second_rot_csv   = glob(root_folder + '/mocap_data/*_rot*.csv')[0]
            second_pos_csv   = glob(root_folder + '/mocap_data/*_*pos.csv')[0]
            second_pos, second_rot, col_names = load_csv_data(second_rot_csv, second_pos_csv)
            mocap_time = second_rot[:, 0]
        except Exception:
            print('=======================')
            print('No second person data!!')
            print('=======================')
            second_data = False

    if not first_data and not second_data:
        print("No mocap data in './mocap_data/'!!!")
        exit(0)
    
    lidar_keytime = dataset_params['lidar_sync'][0]
    if type(lidar_keytime) == int:
        lidar_keytime = lidar_time[lidar_keytime]

    mocap_keytime = dataset_params['mocap_sync'][0]
    if type(mocap_keytime) == int:
        mocap_keytime = mocap_time[mocap_keytime]
    
    # 1. sync. data
    sync_lidarid, sync_mocapid = sync_lidar_mocap(lidar_time, mocap_time, lidar_keytime, mocap_keytime)
    lidar_keyid = np.where(abs(lidar_time - lidar_keytime) < 1/lidar_framerate)[0][0]
    mocap_keyid = np.where(abs(mocap_time - mocap_keytime) < 1/dataset_params['mocap_framerate'])[0][0]

    # 2 choose start time
    if 'start' in dataset_params:
        start =  dataset_params['start']
    elif start == 0:
        start = int(lidar_keyid) - lidar_framerate * 2
    else:
        start = min(int(lidar_keyid) - lidar_framerate * 2, start) 

    start = sync_lidarid.index(start) if start in sync_lidarid else 0
    end   = sync_lidarid.index(end) if end in sync_lidarid else len(sync_lidarid) - 1

    sync_lidarid = sync_lidarid[start:end+1]
    sync_mocapid = sync_mocapid[start:end+1]

    lidar_params = {"framerate": lidar_framerate,
                    "Key frame": int(lidar_keyid),
                    "Key time 1": lidar_keytime,
                    "Key time 2": dataset_params['lidar_sync'][-1],
                    "Relative time": f"{dataset_params['lidar_sync'][-1] - lidar_keytime:.3f}",
                    "Start frame": sync_lidarid[0],
                    "End frame": sync_lidarid[-1],
                    "Total frames": len(lidar_time)}
    
    mocap_params = {"framerate": dataset_params['mocap_framerate'],
                    "Key frame": int(mocap_keyid),
                    "Key time 1": mocap_keytime,
                    "Key time 2": dataset_params['mocap_sync'][-1],
                    "Relative time": f"{dataset_params['mocap_sync'][-1] - mocap_keytime:.3f}",
                    "Start frame": int(sync_mocapid[0]),
                    "End frame": int(sync_mocapid[-1]),
                    "Total frames": len(mocap_time)}
    
    dataset_params['lidar'] = lidar_params
    dataset_params['mocap'] = mocap_params
    save_json_file(json_path, dataset_params)
    
    # 构建表格内容的列表
    table_rows = []
    for param in lidar_params:
        row = [param, lidar_params[param], mocap_params[param]]
        table_rows.append(row)
    
    print_table("Synchronization Parameters", ["Parameter", "LiDAR", "Mocap(IMU)"], table_rows)

    if sync_lidarid[0] > 1000:
        print("Note that the starting frame for the LiDAR is > 1000.")
        print("It is recommended to manually set the starting frame number.")

    if first_data:
        sync_pose, _  = mocap_to_smpl_axis(first_rot[sync_mocapid], 
                                            fix_orit = False, 
                                            col_name=col_names)
        # 2. save synced mocap trans
        trans = save_trajs(save_dir, sync_pose,
                                first_pos[sync_mocapid], sync_mocapid)

        joints, verts    = poses_to_joints(sync_pose[:1], return_verts=True) 
        feet_center      = (joints[0, 7] + joints[0, 8])/2
        feet_center[2]   = verts[..., 2].min()
        trans           -= trans[0] + feet_center  # make the first frame at the origin

        # 3. save synced data
        betas = np.array([0.0] * 10)
        if 'betas' in dataset_params['first_person']:
            betas = dataset_params['first_person']['betas']
            
        if 'gender' in dataset_params['first_person']:
            gender = dataset_params['first_person']['gender']
        else:
            gender = 'male'

        if 'first_person' not in save_data:
            save_data['first_person'] = {}

        save_data['first_person'].update({
                                    'beta': betas, 
                                    'gender': gender, 
                                    'pose': sync_pose, 
                                    'mocap_trans': trans})

    if second_data:
        fix_orit = True if not first_data else False

        sync_pose_2nd, delta_r = mocap_to_smpl_axis(second_rot[sync_mocapid], 
                                           fix_orit=fix_orit,
                                           col_name=col_names)

        second_trans = save_trajs(save_dir, sync_pose_2nd, 
                                  second_pos[sync_mocapid] @ delta_r.T, 
                                  sync_mocapid, 'second')

        betas = np.array([0.0] * 10)
        if 'betas' in dataset_params['second_person']:
            betas = dataset_params['second_person']['betas']

        if 'gender' in dataset_params['second_person']:
            gender = dataset_params['second_person']['gender']
        else:
            gender = 'male'

        if 'second_person' not in save_data:
            save_data['second_person'] = {}
            
        save_data['second_person'].update({
                                    'beta': betas, 
                                    'gender': gender, 
                                    'pose': sync_pose_2nd, 
                                    'mocap_trans':second_trans})
        if not first_data:
            trans = second_trans
            sync_pose = sync_pose_2nd

    save_data['frame_num'] = sync_lidarid
    save_data['framerate'] = lidar_framerate

    with open(param_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {param_file}")
    
    return sync_pose, trans, sync_lidarid, param_file   # type: ignore

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
