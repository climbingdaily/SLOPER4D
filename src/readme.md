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

## **Dependencies**
- FFmpeg (version >= 3.4.11)
- [CSF](https://github.com/jianboqi/CSF) (Optional)
   - Recommend use `pip` to install it in your conda environment
- SMPL
   - Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`, `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl` and `J_regressor_extra.npy` from http://smpl.is.tue.mpg.de and put them in `smpl` directory.

## **Environment**
- Python 3.8.12
- PyTorch 1.7.0
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
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## **Data loader**
coming soon...


## **Data processing**
```bash
root_folder=/path/to/root_folder
```

- ### **Mocap data** 
   1. Convert the `bvh` file to `csv` files
      ```bash
      # pip install bvh-converter 
      bvhconverter -r "/path/to/bvh"
      ```

   2. Jumping peak detection. Used to double check the synchronization time in `dataset_params.json`
      ```bash
      python tools/detect_jumps.py -M "/path/to/bvh" 
      ```

- ### **LiDAR data** 

   1. Scene Mesh reconstruction (With TSDF fusion) and human data generation
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

   2. Human point clouds cropping
      ```bash
      python src/process_human_points.py -R $root_folder [--scene <scene path>]
      ```
- ### **RGB data (for visualization)** 
   Coming soon...


## **Visualization**

- ### **RGB visualization**
   coming soon...


- ### **SMPL visualization**
   Please refer this visualization tool [SMPL-Scene Viewer](https://github.com/climbingdaily/SMPL-Scene-Viewer),
   or [aitviewer](https://github.com/climbingdaily/aitviewer)


## License
The SLOPER4D dataset is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.

