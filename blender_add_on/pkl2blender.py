import os
import sys
import argparse
from copy import deepcopy

import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append('.')
sys.path.append('..')
from smpl import SMPL_Layer

MOCAP_INIT = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

def lidar_to_root(lidar_trans, joints, global_rots, index=15, is_cuda=True):
    """
    Given the lidar translation, the joint positions, and the joint rotations, we can compute the
    translation of the root joint in the lidar frame
    
    Args:
      lidar_trans: the translation of the lidar in the lidar frame
      joints: the joint positions of the skeleton
      global_rots: the global rotation matrix for each joint
      index: the index of the joint to use for the transformation. Defaults to 15
      is_cuda: whether to use GPU or not. Defaults to True
    
    Returns:
      The translation of the root joint in the lidar frame.
    """

    # _, joints, global_rots = poses_to_vertices_torch(pose, batch_size=1024, trans=trans, is_cuda=is_cuda)
    
    root = joints[:, 0]
    joint = joints[:, index]

    matrot = torch.from_numpy(MOCAP_INIT).float()
    if is_cuda:
        matrot = matrot.cuda()
    joint_rots = global_rots[:, index] @ matrot

    # use the first frame's joint positions to lidar postion as the translation
    lidar_to_joint = joint_rots @ joint_rots[0].T @ (joints[0, index] - lidar_trans[0])

    lidar_to_root_trans = lidar_trans + lidar_to_joint + (root - joint)

    return lidar_to_root_trans

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


LIDAR_POS = 15 
def generate_views(position, direction, rad=np.deg2rad(10), dist = 0.2):
    """
    > Given a list of positions and directions, generate a list of views and extrinsics
    
    Args:
      position: the position of the camera
      direction: the direction of the camera. This is a list of 3x3 matrices, where each matrix is the
    rotation matrix of the camera.
      rad: the angle of the camera from the ground
      dist: distance from the camera to the lookat point
    
    Returns:
      view_list is a list of dictionaries, each dictionary contains the trajectory of the camera.
    """
    assert len(position) == len(direction) 

    mocap_init = np.array([[-1, 0, 0, ], [0, 0, 1], [0, 1, 0]])
    base_view = {
        "trajectory" : 
        [
            {
                "field_of_view" : 80.0,
                "front" : [ 0, -np.cos(rad), np.sin(rad)],
                "lookat" : [ 0, 1, 1.7],
                "up" : [ 0, np.sin(rad), np.cos(rad)],
                "zoom" : 0.0065
            }
        ],
    }

    if direction[0].shape[0] == 4:
        func = R.from_quat
    else:
        func = R.from_matrix

    view_list = []
    extrinsic_list = []
    for t, r in zip(position, direction):
        view = deepcopy(base_view)
        rot = func(r).as_matrix()
        rot = R.from_rotvec(-rad * rot[:, 0]).as_matrix() @ rot

        view['trajectory'][0]['front'] = -rot[:, 1]
        view['trajectory'][0]['up'] = rot[:, 2] 
        view['trajectory'][0]['lookat'] = t + rot @ np.array([0, -dist, 0])
        # view_list.append(view)
        
        front = view['trajectory'][0]['front']
        up = view['trajectory'][0]['up']
        origin = view['trajectory'][0]['lookat']

        extrinsic_list.append(view_to_extrinsic(origin, up, front))
    
    return position, extrinsic_list

def view_to_extrinsic(lookat, up, front):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.stack([-np.cross(front, up), -up, -front])
    extrinsic[:3, 3] = - (extrinsic[:3, :3] @ lookat)   # t = - R @ p
    return extrinsic

def extrinsic_to_cam(extrinsic):
    cam = np.eye(4)
    cam[:3, :3] = extrinsic[:3, :3].T @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    cam[:3, 3] = -(extrinsic[:3, :3].T @ extrinsic[:3, 3])
    return cam

def load_point_cloud(pkl_results, person='second_person', points_num = 1024, trans=None):
    if 'point_clouds' not in pkl_results[person]:
        return None
    
    point_clouds = [[]] * len(pkl_results['frame_num'])

    for i, pf in enumerate(pkl_results[person]['point_frame']):
        point_clouds[pkl_results['frame_num'].index(pf)] = pkl_results[person]['point_clouds'][i]

    # pp = np.array([fix_points_num(pts, points_num) for pts in point_clouds])
    return point_clouds

def save_global_vertices(pkl_file):
    # Load the pkl file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Extract the required information
    frame_lenght  = len(data['frame_num'])

    save_vertices = {'first_person': {}, 'second_person': {}}

    smpl_layer = SMPL_Layer(gender='neutral') 
    save_vertices['faces'] = smpl_layer.th_faces.detach().cpu().numpy()
    if 'framerate' in data:
        save_vertices['framerate'] = data['framerate']
    else:
        save_vertices['framerate'] = 20
    save_vertices['point_clouds'] = load_point_cloud(data)
 
    if 'lidar_traj' in data['first_person']:
        lidar_rots = R.from_quat(data['first_person']['lidar_traj'][:, 4:8]).as_matrix()
        position = data['first_person']['lidar_traj'][:, 1:4]
        rotation = lidar_rots @ lidar_rots[0].T
    else:
        rotation = np.stack([np.eye(3)] * frame_lenght)
        position = np.vstack([np.array([0,0,0])] * frame_lenght)

    save_vertices['lidar_extrinsics'] = np.array([np.eye(4)] * len(position))
    
    # lidar_view = generate_views(position, rotation, dist=0, rad=0)
    save_vertices['lidar_extrinsics'][:, :3, :3] = rotation @ np.array([[1,0,0], [0,0,-1], [0,1,0]]) 
    save_vertices['lidar_extrinsics'][:, :3, 3] = position

    def write_vertices(person, pose_attr, trans_attr, attr=None, lidar_trans=None, ):
        if trans_attr not in data[person] or pose_attr not in data[person]:
            print(f'{pose_attr}, {trans_attr} not in {person}')
            return
        
        trans  = data[person][trans_attr]
        pose   = data[person][pose_attr]
        beta   = data[person]['beta'] if 'beta' in data[person]  else np.zeros(10)
        gender = data[person]['gender'] if 'gender' in data[person]  else 'neutral'
        with torch.no_grad():
            vertices, joints, global_rots = poses_to_vertices_torch(pose, trans, 1024, 
                                                betas=torch.tensor([beta]), 
                                                gender=gender, is_cuda=False)
            attr = pose_attr if attr is None else attr
            if lidar_trans is not None:
                offset_lidar_T = lidar_to_root(torch.from_numpy(lidar_trans).float(), joints, global_rots, index=LIDAR_POS, is_cuda=False)
                offset_lidar_T = offset_lidar_T + torch.from_numpy(trans[0]).float() - joints[0, 0]
                vertices = vertices -  torch.from_numpy(trans[:, None, :]).float() + offset_lidar_T[:, None, :]
            save_vertices[person][attr] = vertices.detach().cpu().numpy()
            print(f'{pose_attr}: {len(vertices)} frames is wrote in {person}')

    write_vertices('first_person', 'opt_pose', 'opt_trans')
    write_vertices('first_person', 'pose', 'mocap_trans', 'baseline2', lidar_trans=position)
    write_vertices('first_person', 'pose', 'mocap_trans', 'imu_pose')

    write_vertices('second_person', 'opt_pose', 'opt_trans')
    write_vertices('second_person', 'pose', 'trans', 'baseline2')
    write_vertices('second_person', 'pose', 'mocap_trans', 'imu_pose')



    # Determine the filename for the output file
    pkl_file = os.path.splitext(pkl_file)[0] + f'_vertices.pkl'

    # Save the pkl file
    with open(pkl_file, 'wb') as f:
        pickle.dump(save_vertices, f)

    print(f'File is stored in {pkl_file}')

if __name__ == '__main__':
    # Define argparse arguments
    parser = argparse.ArgumentParser(description='Read a pkl file and save the global vertices in a pkl file')
    parser.add_argument('pkl_file', type=str, help='The name of the pkl file to read')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the given argument
    save_global_vertices(args.pkl_file)
