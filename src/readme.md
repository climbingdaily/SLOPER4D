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


## **Tested Environment**
- Python 3.8, 3.9
- PyTorch >= 1.8.0
- numpy <= 1.23
- Ubuntu 18.04

## **Installation**
```bash
git clone https://github.com/climbingdaily/SLOPER4D.git
cd SLOPER4D
conda create --name sloper4d python==3.9 -y
conda activate sloper4d
pip install -r requirements.txt
pip install --upgrade pyopengl==3.1.4 # ignore the warning
```
## **Dependencies**
- SMPL: Download v1.0.0 SMPL models `SMPL_NEUTRAL.pkl`, `SMPL_FEMALE.pkl` and `SMPL_MALE.pkl` from http://smpl.is.tue.mpg.de and put them in `./smpl/smpl` directory
- (Optional) [FFmpeg](https://ffmpeg.org/download.html) (version >= 3.4.11)
- (Optional) [CSF](https://github.com/jianboqi/CSF): used for ground segmentation
- (Optional) [Metaseg](https://github.com/kadirnar/segment-anything-video): `pip install metaseg`

## **Processing**
```bash
seq_name=seq002_football_001
root_folder=/path/to/sequence_folder
```

- ### **Mocap data** 
   ```bash
   # Convert the `bvh` file to `csv` files
   # pip install bvh-converter 
   bvh-converter -r $root_folder/mocap_data/${seq_name}_second.bvh

   # Jumping peak detection. 
   # Used to double check the synchronization time in `dataset_params.json`
   python utils/detect_jumps.py -M $root_folder/mocap_data/${seq_name}_second.bvh
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
   ```bash
   # Convert the video to images in one sequence: 
   python src/vid2imgs.py $root_folder

   # Convert the video to images for all sequence: 
   bash src/batch_vid2imgs.sh $(dirname "$root_folder")

   # (Optional) pip install metaseg used for mask data generation
   python src/metaseg_demo.py --base_path $root_folder
   ```

## **Data loader**
Here is a testing example of dataloader, you can import the `SLOPER4D_Dataset` class in your code.
```bash
python src/data_loader.py --pkl $root_folder/${seq_name}_labels.pkl
```


## **License**
The SLOPER4D code is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.

