################################################################################
# File: /tool_func.py                                                          #
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

import pandas as pd  
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
import os
import pickle as pkl
import open3d as o3d
import torch
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.split(os.path.abspath( __file__))[0]))

from smpl import SMPL, SMPL_Layer, SMPL_SAMPLE_PLY, COL_NAME
from utils import get_pose_from_bvh, MOCAP_INIT, pypcd

def write_pcd(filepath, data, rgb=None, intensity=None, mode='wb') -> None:
    """
    It takes a point cloud, and writes it to a file on the remote server
    
    Args:
        filepath: The path to the file on the remote server.
        data: numpy array of shape (N, 3)
        rgb: a numpy array of shape (N, 3) value from 0 to 255
        intensity: the intensity of the point cloud, (N, 1) 
        mode: 'w' for write, 'a' for append. Defaults to w
    """

    if rgb is not None and intensity is not None:
        rgb = pypcd.encode_rgb_for_pcl(rgb.astype(np.uint8))
        dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('rgb', np.float32), ('intensity', np.float32)])
        pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], rgb, intensity], dtype=dt)

    elif rgb is not None:
        rgb = pypcd.encode_rgb_for_pcl(rgb.astype(np.uint8))
        dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('rgb', np.float32)])
        pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], rgb], dtype=dt)

    elif intensity is not None:
        dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('intensity', np.float32)])
        pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], intensity], dtype=dt)

    else:
        dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2]], dtype=dt)

    pc = pypcd.PointCloud.from_array(pc)
    with open(filepath, mode = mode) as f: 
        pc.save_pcd_to_fileobj(f, compression='binary')

def read_pcd(pc_pcd):
    # pc_pcd = pypcd.point_cloud_from_path(pcd_file)
    pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
    pc[:, 0] = pc_pcd.pc_data['x']
    pc[:, 1] = pc_pcd.pc_data['y']
    pc[:, 2] = pc_pcd.pc_data['z']
    fields = {}
    count = 3
    if 'rgb' in pc_pcd.fields:
        append = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb'])/255
        pc = np.concatenate((pc, append), axis=1)
        fields['rgb'] = [count, count+1, count+2]
        count += 3
    if 'normal_x' in pc_pcd.fields and 'normal_y' in pc_pcd.fields and 'normal_z' in pc_pcd.fields:        
        append = pc_pcd.pc_data['normal_x'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        append = pc_pcd.pc_data['normal_y'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        append = pc_pcd.pc_data['normal_z'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        fields['normal'] = [count, count+1, count+2]
        count += 3
    if 'intensity' in pc_pcd.fields:        
        append = pc_pcd.pc_data['intensity'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        fields['intensity'] = [count]
        count += 1
    
    return pc, fields

def load_point_cloud(file_name, pointcloud = None, position = None):
    """
    > Load point cloud from local or remote server
    
    Args:
        file_name: the name of the file to be loaded
        pointcloud: The point cloud to be visualized.
        position: the position of the point cloud
    
    Returns:
        A pointcloud object.
    """
    def funcx_x(x):
        y = 155 * np.log2(x/100) / np.log2(864) + 100
        return y
    
    if pointcloud is None:
        pointcloud = o3d.geometry.PointCloud()
        
    if file_name.endswith('.txt'):
        pts = np.loadtxt(file_name)
        xyz = [1,2,3] if pts.shape[1] == 9 else [0,1,2]
        pointcloud.points = o3d.utility.Vector3dVector(pts[:, xyz]) 
    elif file_name.endswith('.pcd'):
        pcd, fields = read_pcd(pypcd.point_cloud_from_path(file_name)) 

        pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
        if 'normal' in fields:
            pointcloud.normals = o3d.utility.Vector3dVector(pcd[:, fields['normal']])

        if 'rgb' in fields:
            pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, fields['rgb']])

        elif 'intensity' in fields:
            
            intensity = pcd[:, fields['intensity']].squeeze()
            if intensity.max() > 255:
                intensity[intensity>100] = funcx_x(intensity[intensity>100])
                
            scale = 1 if intensity.max() < 1.1 else 255
            colors = plt.get_cmap('plasma')(intensity/scale)[:, :3]
            pointcloud.colors = o3d.utility.Vector3dVector(colors)

        if position is not None:
            
            points = np.asarray(pointcloud.points)
            rule1 = abs(points[:, 0] - position[0]) < 40 
            rule2 = abs(points[:, 1] - position[1]) < 40 
            rule3 = abs(points[:, 2] - position[2]) < 20
            rule4 = [(abs(pt - position) > 0.4).any() for pt in points]
            rule = [a and b and c and d for a,b,c,d in zip(rule1, rule2, rule3, rule4)]

            pointcloud = pointcloud.select_by_index(np.arange(len(rule1))[rule]) 
    elif  file_name.endswith('.ply'):
        pointcloud = o3d.io.read_triangle_mesh(file_name)
        pointcloud.compute_vertex_normals()
    else:
        pass
    return pointcloud

def compute_similarity(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    
    Args:
      S1: The first set of points.
      S2: The points in the target image
    
    Returns:
      R, t, scale
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)
    
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    ROT = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(ROT.dot(K)) / var1

    # 6. Recover translation.
    T = mu2 - scale * (ROT.dot(mu1))

    # 7. Error:
    S1_hat = scale * ROT.dot(S1) + T

    if transposed:
        S1_hat = S1_hat.T
    
    if ROT.shape[0] == 2:
        r = np.eye(3)
        r[:2, :2] = ROT
        ROT = r
        t = np.zeros(3)
        t[:2] = T[:2, 0]
        T = t

    return ROT, T, scale

def fix_points_num(points: np.array, num_points: int):

    points = points[~np.isnan(points).any(axis=-1)]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    ratio = int(len(pc.points) / num_points + 0.05)
    if ratio > 1:
        pc = pc.uniform_down_sample(ratio)

    points = np.asarray(pc.points)
    origin_num_points = points.shape[0]

    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    else:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res

def load_csv_data(rotpath, pospath):
    pos_data_csv=pd.read_csv(pospath, dtype=np.float32)
    rot_data_csv=pd.read_csv(rotpath, dtype=np.float32)

    pos_data = np.asarray(pos_data_csv) /100 # cm -> m
    mocap_length = pos_data.shape[0]
    pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
    rot_data = np.asarray(rot_data_csv) # 度
    return pos_data, rot_data, [c.lower() for c in rot_data_csv.columns]

def sync_lidar_mocap(lidar_time, mocap_time, lidar_key_time, mocap_key_time, mocap_frame_time = 0.01):
    """
    Given the time stamps of the lidar and mocap data, and the time stamps of the key frames of the
    lidar and mocap data, the function returns the indices of the lidar and mocap data that correspond
    to the same time
    
    Args:
      lidar_time: the time stamps of the lidar data
      mocap_time: the time stamps of the mocap data
      lidar_key_time: the timestamp of the lidar frame that you want to sync with the mocap data
      mocap_key_time: the time of the mocap frame that you want to align with the lidar frame
      mocap_frame_time: the time interval between two consecutive mocap frames
    """
    if type(lidar_key_time) == int:
        lidar_key_time = lidar_time[lidar_key_time]
    if type(mocap_key_time) == int:
        mocap_key_time = mocap_time[mocap_key_time]
        
    start_time = lidar_key_time - mocap_key_time
    
    # 根据lidar的时间戳，选取对应的mocap的帧
    _lidar_time = lidar_time - start_time
    _lidar_id = []
    _mocap_id = []

    for i, t in enumerate(_lidar_time):
        tt = abs(mocap_time - t) - mocap_frame_time/2
        if tt.min() <= 1e-4:
            _lidar_id.append(i)
            _mocap_id.append(np.argmin(tt))
    
    return _lidar_id, _mocap_id

def segment_plane(pointcloud, return_seg = False, planes=4, dist_thresh = 0.1, return_model = False):
    '''> It segments the point cloud into planes, and returns the point cloud that is not a plane
    
    Parameters
    ----------
    pointcloud
        The point cloud to be segmented.
    return_seg, optional
        if True, return the non-ground pointcloud, else return the ground pointcloud
    planes, optional
        the number of planes to segment.
    
    '''
    # pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=20))
    # pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    colors = np.asarray(pointcloud.colors)
    points = np.asarray(pointcloud.points)
    # normals = np.asarray(pointcloud.normals)

    rest_idx = set(np.arange(len(points)))
    plane_idx = set()
    temp_idx = set()
    
    temp_cloud = o3d.geometry.PointCloud()
    equation_list = []
    for i in range(planes):
        if i == 0:
            plane_model, inliers = pointcloud.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1200)
        elif len(temp_cloud.points) > 300:
            plane_model, inliers = temp_cloud.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1200)
        else:
            break
        # plane_inds += rest_idx[inliers]
        origin_inline = np.array(list(rest_idx))[inliers]
        # colors[origin_inline] = plt.get_cmap('tab10')(i)[:3]
        rest_idx -= set(origin_inline)
        plane_idx = plane_idx.union(set(origin_inline))

        if i == 0:
            temp_cloud = pointcloud.select_by_index(inliers, invert=True)
        else:
            temp_cloud = temp_cloud.select_by_index(inliers, invert=True)

        equation = plane_model[:3] ** 2
        plane_dgree = np.arctan2(equation[2], equation[0] + equation[1]) * 180  / np.pi
        # print(f'Plane degree is {plane_dgree}')
        if plane_dgree < 85: 
            # 如果平面与地面的夹角大于5°
            # colors[origin_inline] = [1, 0, 0]
            plane_idx -= set(origin_inline)
            temp_idx = temp_idx.union(set(origin_inline))
        else:
            equation_list.append(plane_model)


    if return_seg:
        non_ground_idx = np.array(list(rest_idx.union(temp_idx)))
        pointcloud= pointcloud.select_by_index(non_ground_idx)
    else:
        ground_idx = np.array(list(plane_idx))
        pointcloud= pointcloud.select_by_index(ground_idx)
    
    if return_model:
        return pointcloud, equation_list
    else:
        return pointcloud

def poses_to_joints(poses, trans=None, is_cuda=True, batch_size=128, return_verts=False):
    """
    It takes in a batch of poses and returns a batch of joint locations
    
    Args:
      poses: the pose parameters of the SMPL model.
      trans: the translation of the model.
      is_cuda: whether to use GPU or not. Defaults to True
      batch_size: The number of SMPL models to process at once. Defaults to 128
      return_verts: if True, return the vertices of the SMPL model as well as the joints. Defaults to
    False
    
    Returns:
      the joints of the SMPL model.
    """
    poses = poses.astype(np.float32)
    joints = np.zeros((0, 24, 3))

    n = len(poses)
    if is_cuda:
        smpl = SMPL().cuda()
    else:
        smpl = SMPL()
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        if is_cuda:
            cur_vertices = smpl(torch.from_numpy(
                poses[lb:ub]).cuda(), torch.zeros((cur_n, 10)).cuda())
        else:
            cur_vertices = smpl(torch.from_numpy(
                poses[lb:ub]), torch.zeros((cur_n, 10)))
        cur_joints = smpl.get_full_joints(cur_vertices)
        joints = np.concatenate((joints, cur_joints.cpu().numpy()))

    if return_verts:
        if trans is not None:
            trans = trans.astype(np.float32)
            cur_vertices += np.expand_dims(trans, 1) # type: ignore
        return joints, cur_vertices.cpu().numpy() # type: ignore
    else:
        return joints

mesh = o3d.io.read_triangle_mesh(SMPL_SAMPLE_PLY) # a ramdon SMPL mesh
def save_ply(vertice, out_file):
    if type(vertice) == torch.Tensor:
        vertice = vertice.squeeze().cpu().detach().numpy()
    if vertice.ndim == 3:
        assert vertice.shape[0] == 1
        vertice = vertice.squeeze(0)
    mesh.vertices = o3d.utility.Vector3dVector(vertice)
    mesh.vertex_normals = o3d.utility.Vector3dVector()
    mesh.triangle_normals = o3d.utility.Vector3dVector()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_file, mesh)

def load_scene(root_folder, data_name):
    scene_file = os.path.join(root_folder, 'Scenes', data_name + '.pcd')
    print('Loading scene point cloud in: ', scene_file)
    normals_file = os.path.join(root_folder, 'Scenes', data_name + '_normals.pkl')   
    # sub_scene_file = os.path.join(root_folder, data_name + '_sub_scene.pcd')
    # sub_normals_file = os.path.join(fileroot_folderpath, data_name + '_sub_scene_normals.txt')   

    scene_point_cloud = o3d.io.read_point_cloud(scene_file)
    # points = np.asarray(scene_point_cloud.points)
    
    kdtree = o3d.geometry.KDTreeFlann(scene_point_cloud)
    if not os.path.exists(normals_file):
        print('Estimating normals...')
        scene_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=40))
        normals = np.asarray(scene_point_cloud.normals)
        with open(normals_file, 'wb') as f:
            pkl.dump(normals, f)
        print('Save scene normals in: ', normals_file)
    else:
        with open(normals_file, 'rb') as f:
            normals = pkl.load(f)
        scene_point_cloud.normals = o3d.utility.Vector3dVector(normals)

    print(scene_point_cloud)
    return scene_point_cloud, kdtree

def poses_to_vertices_torch(poses, trans, batch_size = 128, betas=torch.zeros(10), gender='male', is_cuda=True):
    """
    It takes in a batch of poses, translations, and betas, and returns a batch of vertices, joints, and
    global rotations
    
    Args:
      poses: (N, 24, 3)
      trans: translation vector
      batch_size: the number of poses to process at once. Defaults to 128
      betas: shape (10,)
      gender: 'male' or 'female' or 'neutral'. Defaults to male
      is_cuda: whether to use GPU or not. Defaults to True
    
    Returns:
      vertices, joints, global_rots
    """
    if type(poses) != torch.Tensor:
        poses = torch.from_numpy(poses.astype(np.float32))
    if type(trans) != torch.Tensor:
        trans = torch.from_numpy(trans.astype(np.float32))
    if type(betas) != torch.Tensor:
        betas = torch.from_numpy(betas.astype(np.float32))

    vertices = torch.zeros((0, 6890, 3))
    joints = torch.zeros((0, 24, 3))
    global_rots = torch.zeros((0, 24, 3, 3))

    smpl = SMPL_Layer(gender=gender)

    if is_cuda:
        smpl = smpl.cuda()
        vertices = vertices.cuda()
        joints = joints.cuda()
        global_rots = global_rots.cuda()
        
        poses = poses.cuda()
        trans = trans.cuda()
        betas = betas.cuda()

    # batch_size = 128
    n = len(poses)
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size
        ub = min(ub, n)

        cur_vertices, cur_jts, cur_rots = smpl(poses[lb:ub], trans[lb:ub], betas)

        vertices = torch.cat((vertices, cur_vertices))
        joints = torch.cat((joints, cur_jts))
        global_rots = torch.cat((global_rots, cur_rots))

    return vertices, joints, global_rots

def poses_to_vertices(poses, batch_size = 128, trans=None, is_cuda=True):
    """
    It takes in a batch of poses, and returns a batch of vertices
    
    Args:
      poses: a numpy array of shape (n, 72)
      batch_size: the number of images to process at once. Defaults to 128
      trans: translation of the model
      is_cuda: whether to use GPU or not. Defaults to True
    
    Returns:
      The vertices of the mesh
    """
    if type(poses) != torch.Tensor:
        poses = poses.astype(np.float32)

    vertices = np.zeros((0, 6890, 3))

    with torch.no_grad():
        n = len(poses)
        smpl = SMPL()
        if is_cuda:
            smpl = smpl.cuda()
        # batch_size = 128
        n_batch = (n + batch_size - 1) // batch_size

        for i in range(n_batch):
            lb = i * batch_size
            ub = (i + 1) * batch_size

            cur_n = min(ub - lb, n - lb)
            if is_cuda:
                cur_vertices = smpl(torch.from_numpy(
                    poses[lb:ub]).cuda(), torch.zeros((cur_n, 10)).cuda())
            else:
                cur_vertices = smpl(torch.from_numpy(
                    poses[lb:ub]), torch.zeros((cur_n, 10)))
            vertices = np.concatenate((vertices, cur_vertices.cpu().numpy()))
        if trans is not None:
            trans = trans.astype(np.float32)
            vertices += np.expand_dims(trans, 1)
    return vertices

def save_json_file(file_name, save_dict):
    """
    Saves a dictionary into json file
    Args:
        file_name:
        save_dict:
    Returns:
    """
    with open(file_name + '_bak.json', 'w') as fp:
        try:
            json.dump(save_dict, fp, indent=4)
            with open(file_name, 'w') as fp:
                json.dump(save_dict, fp, indent=4)
        except Exception as e:
            print(f'{file_name} {e}')
            exit(0)
            
    os.remove(file_name + '_bak.json')
        
def mocap_to_smpl_axis(mocap_rots, fix_orit = False, col_name=COL_NAME):
    """
    > It converts the rotation matrix of each joint in the mocap data to the rotation vector of each
    joint in the SMPL model
    
    Args:
      mocap_rots: the rotation matrix of the mocap data
      fix_orit: whether to fix the orientation of the first frame. Defaults to False
      col_name: the name of the columns in the csv file.
    """
    from tqdm import tqdm
    col_name = [c.lower() for c in col_name]
    new_rot_csv = pd.DataFrame(mocap_rots, columns = col_name)
    mocap_smpl_rots = np.empty(shape=(0, 72))
    
    for count in tqdm(range(mocap_rots.shape[0]), desc='mocap to smpl axis'):
        pp = get_pose_from_bvh(new_rot_csv, count, False).reshape(1,-1)
        mocap_smpl_rots = np.concatenate((mocap_smpl_rots, pp))

    if fix_orit:
        delta_r = R.from_rotvec(mocap_smpl_rots[0,:3]).inv()  #使得第一帧永远朝前
        delta_r = R.from_euler('y', delta_r.as_euler('yxz')[0])
        degrees = delta_r.as_euler('yxz', degrees=True)[0]
        print(f'[Auto] Fix orientation error: Yaw {degrees:.1f} degrees.')
    else:
        delta_r = R.from_matrix(np.eye(3))
        
    mocap_smpl_rots[:,:3] = (R.from_matrix(MOCAP_INIT) * delta_r * R.from_rotvec(mocap_smpl_rots[:,:3])).as_rotvec()

    return mocap_smpl_rots, delta_r.as_matrix()

def read_json_file(file_name):
    """
    Reads a json file
    Args:
        file_name:
    Returns:
    """
    with open(file_name) as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data

def compute_traj_params(mocap, lidar, init_params = None):
    mocap_traj_length = np.linalg.norm(mocap[1:] - mocap[:-1], axis=1).sum()
    lidar_traj_length = np.linalg.norm(lidar[1:] - lidar[:-1], axis=1).sum()

    mocap_XY_length = np.linalg.norm(mocap[1:, :2] - mocap[:-1, :2], axis=1).sum()
    lidar_XY_length = np.linalg.norm(lidar[1:, :2] - lidar[:-1, :2], axis=1).sum()

    mocap_Z_length = abs(mocap[1:, 2] - mocap[:-1, 2]).sum()
    lidar_Z_length = abs(lidar[1:, 2] - lidar[:-1, 2]).sum()

    print(f'Mocap traj lenght: {mocap_traj_length:.3f} m')
    print(f'LiDAR traj lenght: {lidar_traj_length:.3f} m')
    print(f'Mocap XY lenght: {mocap_XY_length:.3f} m')
    print(f'LiDAR XY lenght: {lidar_XY_length:.3f} m')
    print(f'Mocap Z lenght: {mocap_Z_length:.3f} m')
    print(f'LiDAR Z lenght: {lidar_Z_length:.3f} m')
    
    if init_params:
        init_params['lidar_traj_length'] = lidar_traj_length
        init_params['mocap_traj_length'] = mocap_traj_length
        init_params['lidar_XY_length'] = lidar_XY_length
        init_params['mocap_XY_length'] = mocap_XY_length
        init_params['lidar_Z_height'] = lidar_Z_length
        init_params['mocap_Z_heitht'] = mocap_Z_length

    return mocap_traj_length, lidar_traj_length, mocap_XY_length, lidar_XY_length, init_params

class erase_background:

    kdtree = None

    @ staticmethod
    def run(pt, epsilon = 0.10):
        """
        > It takes in a point cloud, and returns a point cloud with the background points removed
        
        :param pt: the point cloud to be processed
        :param epsilon: the distance threshold for the background points
        :return: The remained points after the background is removed.
        """
        remained_index = []
        
        eps2 = epsilon * epsilon
        # kdtree_version
        for idx, p in enumerate(pt.points):
            [_, _, dist_square] = erase_background.kdtree.search_knn_vector_3d(p, 1)
            remained_index.append(idx) if dist_square[0] > eps2 else None
        remained_points = pt.select_by_index(remained_index)
        
        # cloud_to_cloud version
        # too slow
        # dists= np.asarray(pt.compute_point_cloud_distance(erase_background.bg_points))
        # valid_list = np.arange(len(pt.points))[dists > epsilon]
        # remained_points = pt.select_by_index(sorted(valid_list))
        
        return remained_points

def filterTraj(lidar_xyz, frame_id, frame_time=0.05, dist_thresh=0.03, save_type='a', time_interp=False, times=None, fit_time=None):
    """
    The function takes in a trajectory and returns a filtered trajectory
    
    Args:
      lidar_xyz: the lidar data, which is a numpy array with shape (n, 8), where n is the number of
    lidar points. The last column is the timestamp.
      frame_id: the frame number of the lidar point
      frame_time: the time between two frames
      dist_thresh: the distance threshold for the filter. If the distance between the original point and
    the fitted point is less than this threshold, the original point is kept.
      save_type: 'a' or 'b'. Defaults to a
      time_interp: whether to interpolate the time of the trajectory. Defaults to False
      times: the time of each point in the trajectory
      fit_time: the time of the fitted trajectory
    """

    if times is None:
        times = lidar_xyz[:, -1].astype(np.float64)

    if fit_time is None:
        time_gap = times[1:] - times[:-1]
        fit_list = np.arange(
            times.shape[0] - 1)[time_gap > frame_time * 1.5]  # 找到间隔超过1.5倍的单帧时间
        fit_time = []
        for i, t in enumerate(times):
            fit_time.append(t)
            if i in fit_list:
                for j in range(int(time_gap[i]//frame_time)):
                    if time_gap[i] - j * frame_time >= frame_time * 1.5:
                        fit_time.append(fit_time[-1] + frame_time)

        fit_time = np.asarray(fit_time)
        
    # R_quats = R.from_quat(lidar_xyz[:, 4: 8])
    # quats = R_quats.as_quat()
    # spline = RotationSpline(times, R_quats)
    # quats_plot = spline(fit_time).as_quat()
    
    segment = int(1/frame_time)
    trajs = lidar_xyz[:, :3]  # 待拟合轨迹
    trajs_plot = []  # 拟合后轨迹
    length = lidar_xyz.shape[0]
    for i in range(0, length, segment):
        s = i-1   # start index
        e = i+segment   # end index
        if length < e:
            s = length - segment
            e = length
        if s < 0:
            s = 0

        ps = s - segment//2  # filter start index
        pe = e + segment//2  # # filter end index
        if ps < 0:
            ps = 0
            pe += segment//2
        if pe > length:
            ps -= segment//2
            pe = length

        fp = np.polyfit(times[ps:pe],
                        trajs[ps:pe], 3)  # 分段拟合轨迹
        if s == 0:
            fs = np.where(fit_time == times[0])[0][0]  # 拟合轨迹到起始坐标
        else:
            fs = np.where(fit_time == times[i - 1])[0][0]  # 拟合轨迹到起始坐标

        fe = np.where(fit_time == times[e-1])[0][0]  # 拟合轨迹到结束坐标

        if e == length:
            fe += 1
        for j in fit_time[fs: fe]:
            trajs_plot.append(np.polyval(fp, j))

    trajs_plot = np.asarray(trajs_plot)
    save_frame_id = -1 * np.ones(trajs_plot.shape[0]).astype(np.int32).reshape(-1, 1)
    valid_idx = []

    for i, t in enumerate(times):
        old_id = np.where(fit_time == t)[0][0]
        if np.linalg.norm(trajs_plot[old_id] - trajs[i]) < dist_thresh:

            # 拟合值和原始值之间距离若小于阈值，则保留原始值
            trajs_plot[old_id] = trajs[i]
            # quats_plot[old_id] = quats[i]

            # 对于距离小于阈值的帧，帧号记录为原始帧，其余的帧号标记为-1
            if save_type == 'a':
                save_frame_id[old_id] = frame_id[i]
                valid_idx.append(old_id)

        # 所有的点都记录原始的帧号，无标记为-1的帧
        if save_type == 'b':
            save_frame_id[old_id] = frame_id[i]
            valid_idx.append(old_id)


    if time_interp:
        interp_idx = np.where(save_frame_id == -1)[0] # 离群点的ID
        save_frame_id[:,0] = np.arange(trajs_plot.shape[0]) # 帧号重新按照从0开始排列
        valid_idx = np.arange(trajs_plot.shape[0]).astype(np.int64)
    else:
        interp_idx = np.where(save_frame_id[valid_idx] == -1)[0] # 离群点的行号ID

    fitLidar = np.concatenate(
        (save_frame_id[valid_idx], trajs_plot[valid_idx], fit_time[valid_idx].reshape(-1, 1)), axis=1)

    # 4. 保存轨迹
    # save_file = save_in_same_dir(lidar_file, fitLidar, '_filt')  # 保存有效轨迹
    # np.savetxt(save_file.split('.')[0] + '_lineID.txt', interp_idx, fmt='%d')
    return trajs_plot[valid_idx], save_frame_id[valid_idx]