import os
import argparse

import pickle
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

def get_bool_from_coordinates(coordinates, shape=(1080, 1920)):
    """
    This function takes in a set of coordinates and returns a boolean array with True values at those
    coordinates.
    
    Args:
      coordinates: The `coordinates` parameter is a numpy array containing the coordinates of points
    that need to be set to `True` in the boolean array. The first column of the array contains the row
    indices and the second column contains the column indices.
      shape: The shape parameter is a tuple representing the dimensions of the boolean array that will
    be created. In this case, it is set to (1080, 1920), which means the boolean array will have 1080
    rows and 1920 columns.
    
    Returns:
      a boolean array of shape `(1080, 1920)` where the
    elements corresponding to the coordinates in the input `coordinates` are set to `True`. If
    `coordinates` is an empty array, the function returns a boolean array of all `False` values.
    """
    bool_arr = np.zeros(shape, dtype=bool)
    if len(coordinates) > 0:
        bool_arr[coordinates[:, 0], coordinates[:, 1]] = True

    return bool_arr

def fix_points_num(points: np.array, num_points: int):
    """
    The function takes in a numpy array of points and a desired number of points, downsamples the points
    using voxel and uniform downsampling, and either repeats or randomly selects points to reach the
    desired number.
    
    Args:
      points (np.array): `points` is a numpy array containing 3D points.
      num_points (int): num_points is an integer that represents the desired number of points in the
    output point cloud.
    
    Returns:
      a numpy array of shape `(num_points, 3)` containing the
    down-sampled or repeated points from the input `points` array. 
    """
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

class SLOPER4D_Dataset(Dataset):
    def __init__(self, pkl_file, 
                 device='cpu', 
                 return_torch=True, 
                 fix_pts_num=False,
                 print_info=True):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        self.device       = device
        self.return_torch = return_torch
        self.print_info   = print_info
        self.fix_pts_num  = fix_pts_num
        self.lidar_fps    = data['LiDAR_info']['fps'] # scalar
        self.smpl_fps     = data['SMPL_info']['fps']  # scalar
        self.smpl_gender  = data['SMPL_info']['gender']  # string

        self.cam = data['RGB_info']     # 'fps', 'width', 'height', 'intrinsics', 'lidar2cam'(extrinsics), 'dist'

        self.file_basename = data['RGB_frames']['file_basename'] # list of n strings
        self.bbox          = data['RGB_frames']['bbox']          # n x 4 array of scalars
        self.skel_2d       = data['RGB_frames']['skel_2d']       # n x 51 array of scalars
        self.cam_pose      = data['RGB_frames']['cam_pose']      # n x (4, 4)  np.float64, world to camera
        self.extrinsic     = data['RGB_frames']['extrinsic']     # n x 16 array of scalars
        self.tstamp        = data['RGB_frames']['tstamp']        # n x 1 array of scalars
        self.lidar_tstamps = data['RGB_frames']['lidar_tstamps'] # n x 1 array of scalars
        self.smpl_pose     = data['RGB_frames']['smpl_pose']     # n x 216 array of scalars
        self.global_trans  = data['RGB_frames']['global_trans']  # n x 3 array of scalars
        self.betas         = data['RGB_frames']['beta']          # n x 10 array of scalars
        self.human_points  = data['RGB_frames']['human_points']  # list of n arrays, each of shape (x_i, 3)

        self.length = len(self.file_basename)

        mask_pkl = pkl_file[:-4] + "_mask.pkl"
        if os.path.exists(mask_pkl):
            with open(mask_pkl, 'rb') as f:
                print(f'Loading: {mask_pkl}')
                self.masks = pickle.load(f)['masks']
        else:
            self.masks = [[]]*self.length

        self.check_lenght()

        self.data = data
        self.pkl_file = pkl_file

    def updata_pkl(self, img_name, 
                   bbox=None, 
                   cam_pose=None, 
                   keypoints=None):
        if img_name in self.file_basename:
            index = self.file_basename.index(img_name)
            if bbox is not None:
                self.data['RGB_frames']['bbox'][index] = bbox
            if keypoints is not None:
                self.data['RGB_frames']['skel_2d'][index] = keypoints
            if cam_pose is not None:
                self.data['RGB_frames']['cam_pose'][index] = cam_pose
        else:
            print(f"{img_name} is not in the synchronized labels file")

    def save_pkl(self):
        save_path = self.pkl_file[:-4] + '_updated.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"{save_path} saved")

    def check_lenght(self):
        # Check if all the lists inside rgb_frames have the same length
        assert all(len(lst) == self.length for lst in [self.bbox, self.skel_2d, self.extrinsic, 
                                                       self.tstamp, self.lidar_tstamps, self.masks, 
                                                       self.smpl_pose, self.global_trans, 
                                                       self.betas, self.human_points])

        print(f'Data lenght: {self.length}')
        
    def return_smpl_verts(self, is_cuda=False):
        from utils import poses_to_vertices_torch
        with torch.no_grad():
            smpl_verts, joints, global_rots = poses_to_vertices_torch(
                torch.tensor(self.smpl_pose), 
                torch.tensor(self.global_trans), 
                betas=torch.tensor(self.betas)[0:1], 
                gender=self.smpl_gender,
                is_cuda=is_cuda)
            
            return smpl_verts, joints, global_rots
            
    def __getitem__(self, index):
        sample = {
            'file_basename': self.file_basename[index],     
            'lidar_tstamps': self.lidar_tstamps[index],

            'bbox'         : self.bbox[index],
            'mask'         : get_bool_from_coordinates(self.masks[index]),
            'skel_2d'      : self.skel_2d[index],
            'cam_pose'     : self.cam_pose[index],    

            'smpl_pose'    : torch.tensor(self.smpl_pose[index]).float().to(self.device),
            'global_trans' : torch.tensor(self.global_trans[index]).float().to(self.device),
            'betas'        : torch.tensor(self.betas[index]).float().to(self.device),

            'human_points' : fix_points_num(self.human_points[index], 1024) if self.fix_pts_num else self.human_points[index],
        }

        if self.return_torch:
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = torch.tensor(v).to(self.device)
                elif type(v) != str and type(v) != torch.Tensor:
                    sample[k] = torch.tensor(v).float().to(self.device)

        mispart = ''
        mispart += 'box ' if len(sample['bbox']) < 1 else ''
        mispart += 'kpt ' if len(sample['skel_2d']) < 1 else ''
        mispart += 'pts ' if len(sample['human_points']) < 1 else ''
           
        if len(mispart) > 0 and self.print_info:
            print(f'Missing {mispart} in: {index} ')

        return sample
    
    def __len__(self):
        return self.length
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--pkl_file', type=str, 
                        default='/wd8t/sloper4d_publish/seq003_street_002/seq003_street_002_labels.pkl', 
                        help='Path to the pkl file')
    parser.add_argument('--batch_size', type=int, default=3, 
                        help='The batch size of the data loader')
    args = parser.parse_args()
    
    dataset = SLOPER4D_Dataset(args.pkl_file, 
                               return_torch=True, 
                               fix_pts_num=True)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    root_folder = os.path.dirname(args.pkl_file)

    for _, sample in enumerate(dataloader):
        for i in range(args.batch_size):
            pcd_name  = f"{sample['lidar_tstamps'][i]:.03f}".replace('.', '_') + '.pcd'
            img_path  = os.path.join(root_folder, 'rgb_data', sample['file_basename'][i])
            pcd_path  = os.path.join(root_folder, 'lidar_data', 'lidar_frames_rot', pcd_name)
            extrinsic = sample['cam_pose'][i]      # 4x4 lidar to camera transformation
            keypoints = sample['skel_2d'][i]       # 2D keypoints, coco17 style

