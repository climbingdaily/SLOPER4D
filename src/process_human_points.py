################################################################################
# File: /generate_data.py                                                      #
# Created Date: Monday June 27th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

import numpy as np
import os
import configargparse
import sys
from scipy.spatial.transform import Rotation as R
from glob import glob
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import functools
import torch

sys.path.append(os.path.dirname(os.path.split(os.path.abspath( __file__))[0]))

from utils import filterTraj, erase_background, multi_func, save_ply, compute_similarity, poses_to_vertices_torch, select_visible_points, icp_mesh2point

def crop_scene(scene, positions, radius=1):
    """
    It takes a point cloud and a list of positions, and returns a point cloud that contains only the
    points that are within a certain radius of the positions
    
    Args:
      scene_pcd: the point cloud of the scene
      positions: the trajectory of the robot, in the form of a list of 3D points.
      radius: the radius of the sphere around the trajectory that we want to crop. Defaults to 1
    
    Returns:
      a point cloud of the ground points.
    """
    trajectory = o3d.geometry.PointCloud()
    # scene = o3d.geometry.PointCloud()
    # scene.points = scene_pcd.vertices
    trajectory.points = o3d.utility.Vector3dVector(positions - np.array([0 , 0, 0.8]))
    dist = np.asarray(scene.compute_point_cloud_distance(trajectory))
    
    valid_list = np.arange(len(scene.points))[dist < radius]
    ground = scene.select_by_index(sorted(valid_list))
    return ground

def save_smpl(save_dir, lidar_id, pose, trans, beta=np.zeros(10), desc='first_person'):
    person_dir = os.path.join(save_dir, desc)
    os.makedirs(person_dir, exist_ok=True)

    pre = '1' if 'first' in desc else '2'

    mesh_list = [os.path.join(
        person_dir, f'{pre}_{frame_id:04d}.ply') for frame_id in lidar_id]
    vertices, _, _ = poses_to_vertices_torch(pose, trans, 1024, betas=torch.tensor([beta]), is_cuda=False)

    multi_func(save_ply, 8, len(vertices), f'{desc} SMPL', False, vertices.cpu().numpy(), mesh_list)

def get_traj_scale(T1, T2, diff_id, thresh=0.05):
    """
    It computes the ratio of the average distance between two consecutive points in the first trajectory
    to the average distance between two consecutive points in the second trajectory
    
    Args:
      T1: the first trajectory
      T2: the trajectory of the robot
      diff_id: the index of the first frame where the two trajectories differ
    
    Returns:
      The scale of the trajectory.
    """
    L1 = np.linalg.norm(
        (T1[diff_id + 1] - T1[diff_id])[:, :2], axis=1)
    L2 = np.linalg.norm((T2[diff_id + 1] -
                        T2[diff_id])[:, :2], axis=1)
    scale = L1[L2 > thresh].mean() / L2[L2 > thresh].mean()
    return scale

def fuse_trans(root_folder, 
                trans, 
                sync_lidar_id, 
                human_pcds=None, 
                second_frame_id=[], 
                frame_rate=20, 
                traj_data=None,
                save_traj=True):
    """
    Given the tracking trajectory and the IMU trajectory, we first compute the similarity transformation
    between them, and then fuse the two trajectories by replacing the tracking trajectory with the tracking trajectory
    
    Args:
      root_folder: the path to the folder containing the data
      human_pcds: the human point clouds
      trans: the IMU trajectory
      sync_lidar_id: the frame id of the lidar frames
      second_frame_id: the frame id of the second frame of the human
      traj_data: the trajectory data of the human, if it exists.
    
    Returns:
      lidar_trans, scaled_trans, bboxes, ROT
    """
    tracking_traj_path = glob(root_folder + '/lidar_data/*tracking_traj.txt')

    if len(tracking_traj_path) > 0 and traj_data is None:
        traj_data = np.loadtxt(tracking_traj_path[0])
        tracking_traj = traj_data[:, :3]
        try:
            second_frame_id = []
            for i, id in enumerate(traj_data[:, 3:4]):
                second_frame_id.append(int(id)) if int(id) in sync_lidar_id else None
            ll = [np.where(traj_data[:, 3]==fid)[0][0] for fid in second_frame_id]
            tracking_traj = tracking_traj[ll]

        except Exception as e:
            print(e.args[0])
    elif traj_data is not None:
        tracking_traj = traj_data[:, :3]
    else:
        raise ValueError(f'No tracking data!')

    save_path = os.path.join(root_folder, 'synced_data', 'second_person_traj_filt.txt')

    re_id_before_filt =[sync_lidar_id.index(id) for id in second_frame_id]

    if len(second_frame_id)/len(sync_lidar_id) > 0.12:
        # not used when the data is too sparse
        tracking_traj, second_frame_id = filterTraj(tracking_traj, 
                                                    second_frame_id, 
                                                    frame_time=1/frame_rate, 
                                                    dist_thresh=0.1, 
                                                    times=np.array(second_frame_id)/frame_rate)

    # diff_id = np.where(np.diff(second_frame_id) == 1)[0]
    # scale = get_traj_scale(tracking_traj, trans[second_frame_id], diff_id)
    # scaled_trans = get_scaled_traj(trans, scale)

    pt = o3d.geometry.PointCloud()
    bounds_xyz = np.zeros_like(trans) + np.array([1,1,2])
    if human_pcds:
        for idx, h in enumerate(human_pcds):
            pt.points = o3d.utility.Vector3dVector(h)
            box = pt.get_axis_aligned_bounding_box()
            bound = box.max_bound - box.min_bound
            bounds_xyz[re_id_before_filt[idx]] = np.stack([max([1,1,1.8][i], bound[i]) for i in range(3)])

    # concat IMU traj and tracking positions
    # =================================================================

    re_id =[sync_lidar_id.index(id) for id in second_frame_id]

    ROT, T, _ = compute_similarity(trans[re_id][:, :2], tracking_traj[:, :2])
    delta_degree = np.linalg.norm(R.from_matrix(ROT).as_rotvec()) * 180 / np.pi
    print(f'[Second person] {delta_degree:.1f}° around Z-axis from the mocap to the real data.')

    orit_trans = (trans @ ROT.T) + T
    # orit_trans -= orit_trans[0] - trans[0] 
    orit_trans += tracking_traj[0, :3] - orit_trans[re_id[0]] # move to the start position

    # =================
    # contact IMU traj 
    # =================
    fused_trans = orit_trans.copy()
    fused_trans[re_id] = tracking_traj[:, :3]
    
    appendix = np.ones(len(fused_trans))[:, None]
    appendix[re_id] = 0
    
    diff = fused_trans - orit_trans

    for i, j in zip([0] + re_id, re_id + [fused_trans.shape[0] - 1]):

        if i==j or j == i+1 and j != fused_trans.shape[0] - 1:
            continue 

        dists = np.linalg.norm(orit_trans[i+1:j+1] - orit_trans[i:j], axis=1) 
        ddiff = (diff[j] - diff[i])

        if j == fused_trans.shape[0] - 1:
            ddiff = 0
        for k in range(j-i+1):
            ratio = dists[:k].sum() / dists.sum() if k > 0 else 0
            fused_trans[i + k] = orit_trans[i+k] + diff[i] + ddiff * ratio
    
    if save_traj:
        save_data = np.concatenate((np.arange(len(fused_trans))[:, None], fused_trans, appendix), axis=1)
        np.savetxt(save_path, save_data, fmt=['%d', '%.6f', '%.6f', '%.6f', '%d'])
        print(f'[Traj save]: {save_path}')

    bboxes = []
    for bound, center in zip(bounds_xyz, fused_trans):
        # b = o3d.geometry.AxisAlignedBoundingBox(center - bound/2, center + bound/2)
        bboxes.append([center - bound/2, center + bound/2])

    return fused_trans, orit_trans, bboxes, ROT

def dbscan(rest, eps=0.25, min_points=20):
    """
    It takes a point cloud, and returns a list of point clouds, where each point cloud is a cluster
    
    Args:
      rest: the point cloud
      eps (float): Density parameter that is used to find neighbouring points.
      min_points: The minimum number of points to form a cluster. Defaults to 20
    
    Returns:
      A list of point clouds, each point cloud is a cluster.
    """
    labels = np.array(rest.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    # if max_label > 0:
    #     print(f"point cloud has {max_label + 1} clusters")
    numbers = [np.where(labels == i)[0] for i in range(max_label + 1)]
    numbers = sorted(numbers, key=lambda x: len(x))
    return [rest.select_by_index(index) for index in numbers]


def check_valid(pts, valid_num=50, min_z=0.6, min_xy=0.3):
    """
    It checks if the point cloud is valid by checking if the number of points is greater than a
    threshold, and if the bounding box of the point cloud is greater than a threshold in the z direction
    and the x and y directions
    
    Args:
      pts: the point cloud to be checked
      valid_num: The minimum number of points in a point cloud to be considered valid. Defaults to 50
      min_z: minimum height of the bounding box
      min_xy: The minimum xy length of the bounding box.
    
    Returns:
      a boolean value.
    """
    if len(pts.points) > valid_num :
        maxb = pts.get_axis_aligned_bounding_box().max_bound
        minb = pts.get_axis_aligned_bounding_box().min_bound
        bbox_length = abs(maxb-minb)

        if bbox_length[-1] < min_z or (bbox_length[1] + bbox_length[0]) < min_xy:
            return False
        else:
            return True
    return False

def crop_and_save(pcd, frame_id, bbox, point_cloud_dir, save_dir):
    """
    It takes in a point cloud, a frame id, a bounding box, and two directories, and returns a cropped
    point cloud and the frame id if the point cloud is valid
    
    Args:
      pcd: the name of the point cloud file
      frame_id: the frame id of the point cloud
      bbox: the bounding box of the object
      point_cloud_dir: the directory where the point clouds are stored
      save_dir: the directory where the cropped point clouds will be saved
    
    Returns:
      The cropped point cloud and the frame id
    """
    pointcloud = o3d.io.read_point_cloud(os.path.join(point_cloud_dir, pcd))
    cropped_dir = point_cloud_dir + '_cropped'
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
        
    box       = o3d.geometry.AxisAlignedBoundingBox(bbox[0], bbox[1])
    index     = box.get_point_indices_within_bounding_box(pointcloud.points)
    rest_idex = set(np.arange(len(pointcloud.points))) - set(index)
    
    p1   = pointcloud.select_by_index(index)
    p1   = erase_background.run(p1, epsilon=0.05)
    rest = pointcloud.select_by_index(list(rest_idex))
    if not os.path.exists(os.path.join(cropped_dir, pcd)):
        o3d.io.write_point_cloud(os.path.join(cropped_dir, pcd), rest)

    if check_valid(p1):
        try:
            p2 = dbscan(p1, 0.3)[-1]
            if not check_valid(p2):
                return
        except Exception as e:
            p2 = p1
        points = np.asarray(p2.points)
        return points, frame_id

def recrop_humans(save_dir, 
                  bboxes, 
                  frame_ids, 
                  ground_file, 
                  point_cloud_dir=None):
    if point_cloud_dir is None:
        point_cloud_dir = glob(os.path.dirname(save_dir) + '/lidar_data/*lidar_frames_rot')[0]
        
    human_trans = np.array([(box[0] + box[1])/2 for box in bboxes])
    ground_near = crop_scene(o3d.io.read_point_cloud(ground_file), human_trans, 2)
    
    print(f'Ground loaded from {ground_file}')
    erase_background.kdtree = o3d.geometry.KDTreeFlann(ground_near)

    pcd_paths = os.listdir(point_cloud_dir)
    pcd_paths = sorted(pcd_paths, key=lambda x: float(x.split('.')[0].replace('_', '.')))
    pcd_paths = [pcd_paths[i] for i in frame_ids]
    save_dir = os.path.join(save_dir, 'second_person')
    os.makedirs(save_dir, exist_ok=True)

    temp_data = []
    assert len(pcd_paths) == len(frame_ids) == len(bboxes)
    
    # for i in tqdm(range(len(pcd_paths))):
    #     temp_data.append(crop_and_save(pcd_paths[i], frame_ids[i], bboxes[i], point_cloud_dir, save_dir))

    temp_data = multi_func(
        functools.partial(crop_and_save, point_cloud_dir=point_cloud_dir, save_dir=save_dir), 
                           4, 
                           len(pcd_paths), 
                           'Re-crop humans', 
                           False, 
                           pcd_paths, 
                           frame_ids, 
                           bboxes)

    valid_data = []
    _ = [valid_data.append(t) if t is not None else False for t in temp_data]
    # save_human_points= np.stack([t[0] for t in valid_data])
    save_human_points= [t[0] for t in valid_data]
    save_human_idx= np.stack([t[1] for t in valid_data])
    
    return save_human_points, save_human_idx

def update_pkl_2nd(save_dir, 
                lidar_trans, 
                human_points, 
                frameids, 
                rot=np.eye(3), 
                trans_2nd=None):
    """
    > This function takes in the lidar points, the human points, the frameids, the rotation matrix, and
    the mocap translation. It then updates the humans_param.pkl file with the new data
    
    Args:
      save_dir: the directory where the data is saved
      lidar_trans: the translation of the lidar in the world frame
      human_points: the point cloud of the human
      frameids: the frame number of the lidar point cloud
      rot: rotation matrix to rotate the human pose
      trans_2nd: the translation of the mocap data in the lidar frame
    
    Returns:
      The pose of the second person
    """
    param_file = os.path.join(save_dir, 'humans_param.pkl')
    with open(param_file, "rb") as f:
        save_data = pickle.load(f)
        print(f"File loaded in {param_file}")

    human_data = save_data['second_person']
    sync_second_pose = human_data['pose']
    sync_second_pose[:, :3] = (R.from_matrix(
        rot) * R.from_rotvec(sync_second_pose[:, :3])).as_rotvec()
    if trans_2nd is not None:
        human_data['mocap_trans'] = trans_2nd
    human_data['pose'] = sync_second_pose
    human_data['trans'] = lidar_trans
    human_data['point_clouds'] = human_points 
    human_data['point_frame'] = frameids

    temp_pt = o3d.geometry.PointCloud()
    temp_pt.points = o3d.utility.Vector3dVector(np.vstack(human_points))
    temp_pt.paint_uniform_color(plt.get_cmap("tab20")(9)[:3])

    o3d.io.write_point_cloud(os.path.join(save_dir, 'all_cropped_humans.pcd'), temp_pt)

    with open(param_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {param_file}")

    return sync_second_pose

def make_bbox(smpl):
    mesh = o3d.geometry.PointCloud()
    bbox = []
    extent = np.array([0.3, 0.3, 0.3])
    for ss in smpl:
        mesh.points = o3d.utility.Vector3dVector(ss)
        bounds = mesh.get_axis_aligned_bounding_box()
        bbox.append([bounds.min_bound - extent, bounds.max_bound + extent])
    return bbox

def main(root_folder, scene_path, iters=2):
    save_dir = os.path.join(root_folder, 'synced_data')
    param_file = os.path.join(save_dir, 'humans_param.pkl')
    os.makedirs(save_dir, exist_ok=True)

    with open(param_file, "rb") as f:
        f = pickle.load(f)
        print(f"File loaded in {param_file}")

    if 'second_person' in f:
        pose_2nd = f['second_person']['pose']
        trans_2nd = f['second_person']['mocap_trans']
        beta = f['second_person']['beta']
        if 'gender' in f['second_person']:
            gender = f['second_person']['gender']
        else:
            gender = 'male'
        
        with torch.no_grad():
            _, joints, _ = poses_to_vertices_torch(pose_2nd, trans_2nd, 1024, betas=torch.tensor([beta]), gender=gender, is_cuda=False)

        root_joints = joints[:, 0].detach().cpu().numpy()

        joints_to_trans_offset = (trans_2nd - root_joints).mean(0)

        fused_roots_joints, \
        orit_roots_joints, \
        bbox, \
        rot = fuse_trans(root_folder, 
                        trans=root_joints, 
                        sync_lidar_id=f['frame_num'], 
                        frame_rate=f['framerate'])
        fused_trans = fused_roots_joints + joints_to_trans_offset
        orit_trans = orit_roots_joints + joints_to_trans_offset
        
        count = 0
        while True:
            pose = f['second_person']['pose'].copy()

            count += 1
            if count > iters:
                break

            pose[:, :3] = (R.from_matrix(rot) * R.from_rotvec(pose[:, :3])).as_rotvec()
            with torch.no_grad():
                smpl, _, _ = poses_to_vertices_torch(pose, fused_trans, 1024, betas=torch.tensor([beta]), gender=gender, is_cuda=False)
            # root_joints = joints[:, 0].detach().cpu().numpy()
            human_points_list, frameid_2nd = recrop_humans(
                save_dir, bbox, f['frame_num'], ground_file=scene_path)
            indexes = [f['frame_num'].index(i) for i in frameid_2nd]

            vis_smpl_idx = multi_func(select_visible_points, 8, 
                                    pose.shape[0], 'select_visible_points', False, 
                                    smpl.cpu().numpy(), 
                                    f['first_person']['lidar_traj'][:, 1:8], 
                                    print_progress=True)

            delta_trans = multi_func(icp_mesh2point, 
                                    8, len(indexes), 'ICP', False,
                                    smpl.cpu().numpy()[indexes],
                                    human_points_list, 
                                    f['first_person']['lidar_traj'][indexes, 1:8], 
                                    vis_smpl_idx)
            
            delta_trans = np.array(delta_trans)
            valid_icp = (np.linalg.norm(delta_trans, axis=-1) > 1e-4)
            print(f'[Valid ICP results]: {sum(valid_icp)}')

            fused_roots_joints, orit_roots_joints, \
                    bbox, rot = fuse_trans(root_folder, 
                                            root_joints, 
                                            f['frame_num'], 
                                            frame_rate=f['framerate'],
                                            second_frame_id=frameid_2nd, 
                                            traj_data=fused_roots_joints[indexes] + delta_trans)
            
            fused_trans = fused_roots_joints + joints_to_trans_offset
            orit_trans = orit_roots_joints + joints_to_trans_offset

        pose[:, :3] = (R.from_matrix(rot) * R.from_rotvec(pose[:, :3])).as_rotvec()
        with torch.no_grad():
            smpl, _, _ = poses_to_vertices_torch(pose, fused_trans, 1024, betas=torch.tensor([beta]), gender=gender, is_cuda=False)

        human_points_list, frameid_2nd = recrop_humans(save_dir, 
                                                       make_bbox(smpl.cpu().numpy()), 
                                                        f['frame_num'],
                                                        ground_file=scene_path)
        print(f'Valid human points frames: {len(human_points_list)}')
        sync_second_pose = update_pkl_2nd(save_dir, 
                                        fused_trans, 
                                        human_points_list, 
                                        frameid_2nd, 
                                        rot, 
                                        orit_trans)
        
    #     save_smpl(save_dir, 
    #             f['frame_num'], 
    #             sync_second_pose, 
    #             fused_trans,
    #             beta,
    #             'second_person')

if __name__ == '__main__':  
    parser = configargparse.ArgumentParser()
                        
    parser.add_argument("--root_folder", '-R', type=str, default='')
    parser.add_argument("--iters", '-I', type=int, default=2)
    parser.add_argument("--scene", '-S', type=str, default=None)
    parser.add_argument("--ground", action='store_true', 
                        help='wether to use the ground file as the the background. not used in multi-floor cases')
    
    args, opts = parser.parse_known_args()
    append = '_ground' if args.ground else ''

    if args.scene is None:
        try:
            scene_path = glob(args.root_folder + f'/lidar_data/*frames{append}.ply')[0]
        except:
            print('No default scene file!!!')
            exit(0)
    else:
        scene_path = os.path.join(args.root_folder, 'lidar_data', args.scene)
        
        if os.path.exists(args.scene):
            scene_path = args.scene
        elif os.path.exists(scene_path):
            pass
        else:
            print('No scene file!!!')
            exit(0)
        
    main(args.root_folder, scene_path, args.iters)
