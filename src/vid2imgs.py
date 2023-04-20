import os
import argparse
from tqdm import tqdm

import cv2

class Vid2imgs:
    def __init__(self, video_path, save_path):
        assert os.path.isfile(video_path), "Error: video no exist!"

        self.vid = cv2.VideoCapture()
        self.vid.open(video_path)
        assert self.vid.isOpened(), "Error: video not opened!"

        self.vid_fps   = float(self.vid.get(cv2.CAP_PROP_FPS))
        self.save_path = save_path
        
        os.makedirs(self.save_path, exist_ok=False)
        print(f"Images will be saved to \n ==> {self.save_path}")

    def _vid_frames(self):
        while self.vid.isOpened():
            success, frame = self.vid.read()
            if success: 
                yield frame
            else: 
                break

    def run(self):
        img_idx = 0
        progress_bar = tqdm(self._vid_frames())
        for frame in progress_bar:

            timestamp = img_idx / self.vid_fps
            img_fname = f"{timestamp:06.06f}.jpg"
            img_path  = os.path.join(self.save_path, img_fname)

            progress_bar.set_description(f"Saving {img_fname}")
            cv2.imwrite(img_path, frame)
            img_idx += 1
            
seqs = [
    'seq001_campus_001',
    'seq002_football_001',
    'seq003_street_002',
    'seq004_library_001',
    'seq005_library_002',
    'seq006_library_003',
    'seq007_garden_001',
    'seq008_running_001',
    'seq009_running_002',
    'seq010_park_001',
    'seq011_park_002',
    'seq012_musicsquare_001',
    'seq013_sunlightrock_001',
    'seq014_garden_002',
    'seq015_mayuehansquare_001'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str)
    return parser.parse_args() 
            
if __name__ == '__main__':
    args        = parse_args()
    root_folder = args.root_folder
    
    for seq in seqs:
        seq_path  = os.path.join(root_folder, seq, 'rgb_data', seq+'.MP4')
        save_path = os.path.join(root_folder, seq, 'rgb_data', seq+'_imgs')
        if not os.path.isfile(seq_path): 
            print("Not found: ", seq_path)
            continue
        if os.path.exists(save_path): 
            print("Make sure not exist: ", save_path)
            continue
        to_images = Vid2imgs(seq_path, save_path)
        to_images.run()
        