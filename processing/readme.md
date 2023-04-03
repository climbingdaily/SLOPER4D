# Data processing pipeline

## **Data sequence structure**
```bash
├── root_folder
   ├── lidar_data/
   |  ├── lidar_frames_rot/        
   |  |   └── '*.pcd'              # undistorted n frames point clouds in global coordinates
   |  ├── 'lidar_trajectory.txt'   # everyline: framenum X Y Z qx qy qz qw timestamp
   |  └── 'tracking_traj.txt'      # everyline: X Y Z framenum timestamp
   ├── mocap_data/
   |  └── '*_second.bvh'           # mocap data
   ├── rgb_data/
   |  ├── 'annotation.pkl'         # video annotations
   |  └── '*.mp4'
   └── 'dataset_params.json'       # some important setup params
```

- Define the variable `$root_folder`
```bash
root_folder=root_folder_path
```

## **Mocap data processing** 
---

   1. Convert the `bvh` file to `csv` files
      ```bash
      # pip install bvh-converter 
      bvhconverter -r "/path/to/bvh"
      ```

      The new generated flies are shown below
      ```bash
      ├── root_folder
         ├── .../
         ├── mocap_data/
         |  ├── '*_second.bvh'       <====
         |  ├── '*_rotations.csv'    <====
         |  └── '*_worldpos.csv'     <====
      ```
   4. (Optional) Jumping peak detection, the image will be saved in the same directory. Used to double check the time in `dataset_params.json`
      ```bash
      python tools/detect_jumps.py --mpcap_file "/path/to/bvh" 
      ```

## **LiDAR data processing** 

- Scene Mesh reconstruction (With TSDF fusion) and human data generation
   ``` bash
   python processing/process_raw_data.py --root_folder $root_folder --tsdf --sync 
   ```

   ``` bash
   ├── root_folder
      ├── lidar_data/
      |  ├── ...
      |  └── '*frames.ply'        <======    # Scene mesh
      ├── synced_data/
      |  └── 'humans_param.pkl'   <======    # human data
   ```
   
   optional arguments:
   ```
  --root_folder         The data's root directory
  --traj_file 
  --params_file 
  -S, --start_idx       The start frame index in LiDAR for processing, specified when sychronization time is too late
  -E, --end_idx         The end frame index in LiDAR for processing, specified when sychronization time is too early.
  -VS, --voxel_size     The voxel filter parameter for TSDF fusion
  --skip_frame          The everay n frame used for mapping
  --tsdf                Use VDB fusion to build the scene mesh 
  --sdf_trunc           The trunction distance for SDF funtion
  --sync                Synced all data and save a pkl based on the params_file
   ```

- Human point clouds cropping
   ```bash
   python processing/process_human_points.py -R $root_folder [--scene <scene path>]
   ```