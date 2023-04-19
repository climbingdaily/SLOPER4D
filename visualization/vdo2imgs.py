import os
import tqdm

import cv2
from utils import *

import argparse



class VdoToImages:
    def __init__(self, video_path):
        assert os.path.isfile(video_path), "Error: video no exist!"
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.vdo = cv2.VideoCapture()
        self.vdo.open(video_path)
        assert self.vdo.isOpened(), "Error: video not opened!"
        self.vdo_fps = float(self.vdo.get(cv2.CAP_PROP_FPS))
        output_path = os.path.dirname(video_path)
        self.img_save_path = os.path.join(output_path, base_name+'_imgs')
        os.makedirs(self.img_save_path, exist_ok=True)

    def _vdo_frames(self):
        while self.vdo.isOpened():
            success, frame = self.vdo.read()
            if success: yield frame
            else: break

    def run(self):
        img_idx = 0
        for im in tqdm.tqdm(self._vdo_frames()):
            img_timestamp = img_idx / self.vdo_fps
            img_idx += 1
            img_fname = "%06.06f.jpg"%(img_timestamp)
            cv2.imwrite(os.path.join(self.img_save_path, img_fname), im)


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
    parser.add_argument("--dataset_path", 
        default='/wd8t/sloper4d_publish', 
        type=str
    )
#     parser.add_argument("--save_path", 
#         type=str, required=True
#     )
            
if __name__ == '__main__':
    args = parse_args()
    dataset_path = args.dataset_path
    for seq in seqs:
        seq_path = os.path.join(dataset_path, seq, 'rgb_data', seq+'.MP4')
        if not os.path.isfile(seq_path): 
            print("Not found: ", seq_path)
            continue
        to_images = VdoToImages(seq_path)
        to_images.run()
        