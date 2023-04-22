import os
import argparse
from tqdm import tqdm

import cv2

class Vid2imgs:
    def __init__(self, video_path):
        assert os.path.isfile(video_path), "Error: video no exist!"

        self.vid = cv2.VideoCapture()
        self.vid.open(video_path)

        assert self.vid.isOpened(), "Error: video not opened!"

        self.vid_fps   = float(self.vid.get(cv2.CAP_PROP_FPS))
        self.save_path = f"{os.path.splitext(video_path)[0]}_imgs"
        
        os.makedirs(self.save_path, exist_ok=True)
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str)
    return parser.parse_args() 
            
if __name__ == '__main__':
    args        = parse_args()
    root_folder = args.root_folder
    
    seq_name    = os.path.basename(root_folder)
    vid_path    = os.path.join(root_folder, 'rgb_data', seq_name+'.MP4')

    if not os.path.isfile(vid_path): 
        print("Not found: ", vid_path)
    else:
        to_images = Vid2imgs(vid_path)
        to_images.run()
        