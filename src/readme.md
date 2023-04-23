# Data processing pipeline


## **Data structure**
```bash
# the data structure for every sequence
├── root_folder
   ├── lidar_data/
   |  ├── lidar_frames_rot/        
   |  |   └── '*.pcd'              # undistorted n frames point clouds in global coordinates
   |  ├── 'lidar_trajectory.txt'   # everyline: framenum X Y Z qx qy qz qw timestamp
   |  └── 'tracking_traj.txt'      # everyline: X Y Z framenum timestamp
   ├── mocap_data/
   |  └── '*_second.bvh'           # mocap data
   ├── rgb_data/
   |  └── '*.mp4'
   ├── '*_labels.pkl'              # all 2D/3D labels and origin human data
   └── 'dataset_params.json'       # meta info
```


## **Environment**
- Python 3.8.12
- PyTorch >= 1.8.0
- CUDA 11.0
- Ubuntu 18.04

## **Installation**
1. Clone the repository:
```bash
git clone https://github.com/climbingdaily/SLOPER4D.git
```
2. Install the required packages:
```bash
conda create --name sloper4d python==3.8 -y
conda activate sloper4d
conda install pytorch==1.8.0 torchvision==0.8.0 torchaudio==0.8.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
## **Dependencies**
- **SMPL**: Download v1.0.0 version SMPL models `SMPL_NEUTRAL.pkl`, `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl` and `J_regressor_extra.npy` from http://smpl.is.tue.mpg.de and put them in `smpl` directory
- Install [detectron2](https://github.com/facebookresearch/detectron2.git) and [pypenGL](https://github.com/mcfletch/pyopengl.git), then update pyopengl to 3.1.4 by `pip install --upgrade pyopengl==3.1.4` for visualization. 
- **FFmpeg** (version >= 3.4.11)
- [**CSF**](https://github.com/jianboqi/CSF): Optional, used for ground segmentation

## **Processing**
```bash
seq_name=seq002_football_001
root_folder=/path/to/sequence_folder
```

- ### **Mocap data** 
   Convert the `bvh` file to `csv` files
   ```bash
   # pip install bvh-converter 
   bvh-converter -r $root_folder/mocap_data/${seq_name}_second.bvh
   ```

   Jumping peak detection. Used to double check the synchronization time in `dataset_params.json`
   ```bash
   python utils/detect_jumps.py -M $root_folder/mocap_data/$seq_name\_second.bvh
   ```

- ### **LiDAR data** 

   Scene Mesh reconstruction (With TSDF fusion) and raw human data generation
   ``` bash
   python src/process_raw_data.py --root_folder $root_folder --tsdf --sync 
   ```
   optional arguments:
   ```
   --root_folder          The data's root directory
   --traj_file  
   --params_file  
   -S, --start_idx        The start frame index in LiDAR for processing, 
                           specified when sychronization time is too late
   -E, --end_idx          The end frame index in LiDAR for processing, 
                           specified when sychronization time is too early.
   -VS, --voxel_size      The voxel filter parameter for TSDF fusion
   --skip_frame           The everay n frame used for mapping
   --tsdf                 Use VDB fusion to build the scene mesh 
   --sdf_trunc            The trunction distance for SDF funtion
   --sync                 Synced all data and save them in a pkl based on the params_file
   ```

   Human point clouds cropping
   ```bash
   python src/process_human_points.py -R $root_folder  
   ```
- ### **RGB data** 
   Convert the video to images in one sequence: 

   ```shell
   python src/vid2imgs.py $root_folder
   ```

   Convert the video to images for all sequence: 
   ```shell
   bash src/batch_vid2imgs.sh $(dirname "$root_folder")
   ```

## **Data loader**
coming soon...



## **License**
The SLOPER4D code is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.

