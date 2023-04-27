import os
import argparse
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class SLOPER4D_Dataset(Dataset):
    def __init__(self, pkl_file, device='cpu', return_torch=True):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        self.device       = device
        self.return_torch = return_torch
        self.lidar_fps    = data['LiDAR_info']['fps'] # scalar
        self.smpl_fps     = data['SMPL_info']['fps']  # scalar
        self.smpl_gender  = data['SMPL_info']['gender']  # string

        self.rgb_fps        = data['RGB_info']['fps']        # scalar
        self.rgb_width      = data['RGB_info']['width']      # scalar
        self.rgb_height     = data['RGB_info']['height']     # scalar
        self.rgb_intrinsics = data['RGB_info']['intrinsics'] # list of 4 scalars
        self.rgb_lidar2cam  = data['RGB_info']['lidar2cam']  # list of 16 scalars
        self.rgb_dist       = data['RGB_info']['dist']       # list of 5 scalars

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
            'tstamp'       : self.tstamp[index],

            'bbox'         : self.bbox[index],
            'mask'         : self.masks[index],
            'skel_2d'      : self.skel_2d[index],
            'cam_pose'     : self.cam_pose[index],    

            'smpl_pose'    : torch.tensor(self.smpl_pose[index]).float().to(self.device),
            'global_trans' : torch.tensor(self.global_trans[index]).float().to(self.device),
            'betas'        : torch.tensor(self.betas[index]).float().to(self.device),

            'human_points' : self.human_points[index],
        }

        if self.return_torch:
            for k, v in sample.items():
                if type(v) != str and type(v) != torch.Tensor:
                    sample[k] = torch.tensor(v).float().to(self.device)

        mispart = ''
        mispart += 'box ' if len(sample['bbox']) < 1 else ''
        mispart += 'kpt ' if len(sample['skel_2d']) < 1 else ''
        mispart += 'pts ' if len(sample['human_points']) < 1 else ''
           
        if len(mispart) > 0:
            print(f'Missing {mispart} in: {index} ')

        return sample
    
    def __len__(self):
        return self.length
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--pkl_file', type=str, default='/wd8t/sloper4d_publish/seq003_street_002/seq003_street_002_labels.pkl', help='Path to the pkl file')
    args = parser.parse_args()
    
    dataset = SLOPER4D_Dataset(args.pkl_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, sample in enumerate(dataloader):
        # print(f'Sample {i}')
        pass

