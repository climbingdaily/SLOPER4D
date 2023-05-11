import os
import json
import pickle

import cv2

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # basics
    parser.add_argument('--base_path', default= '/wd8t/sloper4d_publish',
        type=str, #required=True,
        help='path to dataset'
    )
    parser.add_argument("--scene_name", default='seq005_library_002',
        type=str, #required=True 
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    scene_path = os.path.join(args.base_path, args.scene_name)
    pkl_path = os.path.join(scene_path, args.scene_name+'_labels.pkl')
    data_params_path = os.path.join(scene_path, 'dataset_params.json')
    vdo_path = os.path.join(scene_path, 'rgb_data', args.scene_name+'.MP4')
    
    vid = cv2.VideoCapture()
    vid.open(vdo_path)
    assert vid.isOpened(), "Error: video not opened!"
    vid_fps = float(vid.get(cv2.CAP_PROP_FPS))
    endIdx  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    
    with open(data_params_path, 'r') as f:
        data_params_data = json.load(f)
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    time_rgb2lidar = data_params_data['lidar_sync'][0] - data_params_data['rgb_sync'][0]
    lidar_tstamp = pkl_data['first_person']['lidar_traj'][:, -1]
    time_dev = 1 / vid_fps / 2
    
    lidar_idx = 0
    file_basename = []
    img_devs = []
    for i in range(0, endIdx):
        time_rgb = i/vid_fps + time_rgb2lidar
        if lidar_idx >= lidar_tstamp.shape[0]: break
        # while(lidar_tstamp[lidar_idx] < time_rgb and lidar_tstamp[lidar_idx] - time_rgb > time_dev): lidar_idx += 1
        if abs(lidar_tstamp[lidar_idx]-time_rgb) > time_dev:
            continue
        img_devs.append(abs(lidar_tstamp[lidar_idx]-time_rgb))
        fname = f"{i/vid_fps:.06f}.jpg"
        file_basename.append(fname)
        lidar_idx += 1
    print("num of RGB frames:", len(file_basename))
    print("num of LiDAR frames:", lidar_tstamp.shape[0])
    if len(file_basename)!=lidar_tstamp.shape[0]:
        if lidar_idx < lidar_tstamp.shape[0]: print("RGB end before final lidar frame.")
        else: print("!!!!!ERROR in time alignment!!!!!")
    pkl_data['RGB_frames']['total_frames'] = len(file_basename)
    pkl_data['RGB_frames']['file_basename'] = file_basename
    pkl_data['RGB_info'] = {
        "width":    data_params_data['RGB_info']['width'],
        "height":   data_params_data['RGB_info']['height'],
        "fps":      vid_fps,
        "intrinsics": data_params_data['RGB_info']['intrinsics'],
        "lidar2cam": data_params_data['RGB_info']['lidar2cam'],
        "dist":     data_params_data['RGB_info']['dist']
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        
    print("max offset between lidar and RGB: ", max(img_devs))
        