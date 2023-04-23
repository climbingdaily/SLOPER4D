import argparse
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class SLOPER4D_Dataset(Dataset):
    def __init__(self, pkl_file, device='cpu'):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        self.device    = device
        self.lidar_fps = data['LiDAR_info']['fps'] # scalar
        self.smpl_fps  = data['SMPL_info']['fps']  # scalar
        self.rgb_fps   = data['RGB_info']['fps']   # scalar

        self.rgb_width      = data['RGB_info']['width']      # scalar
        self.rgb_height     = data['RGB_info']['height']     # scalar
        self.rgb_intrinsics = data['RGB_info']['intrinsics'] # list of 4 scalars
        self.rgb_lidar2cam  = data['RGB_info']['lidar2cam']  # list of 16 scalars
        self.rgb_dist       = data['RGB_info']['dist']       # list of 5 scalars

        self.file_basename = data['RGB_frames']['file_basename'] # list of n strings
        self.bbox          = data['RGB_frames']['bbox']          # n x 4 array of scalars
        self.skel_2d       = data['RGB_frames']['skel_2d']       # n x 51 array of scalars
        self.extrinsic     = data['RGB_frames']['extrinsic']     # n x 16 array of scalars
        self.tstamp        = data['RGB_frames']['tstamp']        # n x 1 array of scalars
        self.lidar_tstamps = data['RGB_frames']['lidar_tstamps'] # n x 1 array of scalars
        self.smpl_pose     = data['RGB_frames']['smpl_pose']     # n x 216 array of scalars
        self.global_trans  = data['RGB_frames']['global_trans']  # n x 3 array of scalars
        self.beta          = data['RGB_frames']['beta']          # n x 10 array of scalars
        self.human_points  = data['RGB_frames']['human_points']  # list of n arrays, each of shape (x_i, 3)

        # Check if all the lists inside rgb_frames have the same length
        self.length = len(self.file_basename)

        assert all(len(lst) == self.length for lst in [self.bbox, self.skel_2d, self.extrinsic, 
                                                       self.tstamp, self.lidar_tstamps, 
                                                       self.smpl_pose, self.global_trans, 
                                                       self.beta, self.human_points])
        
        print(f'Data lenght: {self.length}')
        
    def __getitem__(self, index):
        sample = {
            'file_basename': self.file_basename[index],
            'bbox'         : torch.tensor(self.bbox[index]).float().to(self.device),
            'skel_2d'      : torch.tensor(self.skel_2d[index]).float().to(self.device),
            'extrinsic'    : torch.tensor(self.extrinsic[index]).float().to(self.device),
            'tstamp'       : torch.tensor(self.tstamp[index]).float().to(self.device),
            'lidar_tstamps': torch.tensor(self.lidar_tstamps[index]).float().to(self.device),
            'smpl_pose'    : torch.tensor(self.smpl_pose[index]).float().to(self.device),
            'global_trans' : torch.tensor(self.global_trans[index]).float().to(self.device),
            'beta'         : torch.tensor(self.beta[index]).float().to(self.device),
            'human_points' : torch.tensor(self.human_points[index]).float().to(self.device),
        }

        mispart = ''
        mispart += 'box ' if len(sample['bbox']) < 1 else ''
        mispart += 's2d ' if len(sample['skel_2d']) < 1 else ''
        mispart += 'pts ' if len(sample['human_points']) < 1 else ''
           
        if len(mispart) > 0:
            print(f'Missing {mispart} in: {index} ')

        return sample
    
    def __len__(self):
        return self.length
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--pkl_file', type=str, default='/wd8t/sloper4d_publish/seq004_library_001/seq004_library_001_labels.pkl', help='Path to the pkl file')
    args = parser.parse_args()
    
    dataset = SLOPER4D_Dataset(args.pkl_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, sample in enumerate(dataloader):
        # print(f'Sample {i}')
        pass

