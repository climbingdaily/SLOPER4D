import os
import sys

import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm

sys.path.append('.')
sys.path.append('..')
from src import  SLOPER4D_Loader

# pip install metaseg
from metaseg.generator.predictor import SamPredictor
from metaseg.generator.build_sam import sam_model_registry
# from metaseg import sahi_sliced_predict, SahiAutoSegmentation
# from metaseg import SegManualMaskPredictor

from metaseg.utils import (
    download_model,
    multi_boxes,
    save_image,
    show_image,
)

def load_box(box, image, color=(0, 255, 0)):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x, y), (w, h), color, 2)
    return image

def load_image(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_mask(mask, random_color):

    if random_color:
        color = np.random.rand(3) * 255
    else:
        color = np.array([100, 50, 0])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    return mask_image

def load_video(video_path, output_path="output.mp4", output_fps=None):

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = cap.get(cv2.CAP_PROP_FPS) if output_fps is None else output_fps
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    return cap, out

def get_bool_array_from_coordinates(coordinates, shape):
    bool_arr = np.zeros(shape, dtype=bool)

    bool_arr[coordinates[:, 0], coordinates[:, 1]] = True

    return bool_arr

def compress_masks(masks):
    coord = np.transpose(masks.nonzero()).astype(np.uint16)
    return coord

def expand_bbox(left, right, top, bottom, img_width, img_height, ratio=0.1):
    # expand bbox for containing more background

    width = right - left
    height = bottom - top
    # ratio = 0.1 # expand ratio
    new_left = np.clip(left - ratio * width, 0, img_width)
    new_right = np.clip(right + ratio * width, 0, img_width)
    new_top = np.clip(top - ratio * height, 0, img_height)
    new_bottom = np.clip(bottom + ratio * height, 0, img_height)

    return [int(new_left), int(new_right), int(new_top), int(new_bottom)]


def get_box(pose, img_height, img_width, pose_dim=3, ratio=0.1):

    pose = np.array(pose).reshape(-1, pose_dim)
    xmin = np.min(pose[:,0])
    xmax = np.max(pose[:,0])
    ymin = np.min(pose[:,1])
    ymax = np.max(pose[:,1])

    return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height, ratio)


def load_boxes(smpl_masks, boxes, cam, ratio=0.1):
    for i, mask in enumerate(smpl_masks):
        if len(mask) > 200:
            (x1, x2, y1, y2) = get_box(mask, 
                        cam['height'], 
                        cam['width'], 
                        pose_dim=2,
                        ratio=ratio)

            if abs(x2 - x1) > 30 or abs(y2-y1) > 30:
                boxes[i]  = [x1, y1, x2, y2]
    return boxes

class SegManualMaskPredictor:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def image_predict(
        self,
        source,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
        output_path="output.png",
        random_color=False,
        show=False,
        save=False,
    ):
        image = load_image(source)       # RGB format
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        predictor.set_image(image)

        if type(input_box[0]) == list:
            input_boxes, new_boxes = multi_boxes(input_box, predictor, image)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=new_boxes,
                multimask_output=False,
            )
            for mask in masks:
                mask_image = load_mask(mask.cpu().numpy(), random_color)

            for box in input_boxes:
                image = load_box(box.cpu().numpy(), image)

        elif type(input_box[0]) == int:
            input_boxes = np.array(input_box)[None, :]

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_boxes,
                multimask_output=multimask_output,
            )
            mask_image = load_mask(masks, random_color)
            image = load_box(input_box, image)

        combined_mask = cv2.add(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask_image)
        if save:
            save_image(output_path=output_path, output_image=combined_mask)

        if show:
            show_image(combined_mask)

        return masks

    def video_predict(
        self,
        source,
        model_type,
        input_all_boxes=None,
        input_all_point=None,
        input_all_label=None,
        multimask_output=False,
        output_path="output.mp4",
        random_color=False,
        out_fps=None,
        time_stamps=None,
        save_video=True,
        visibel_thresh=0.7,
    ):
        cap, out = load_video(source, output_path, out_fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0
        output_masks = []

        with torch.no_grad():

            for index in tqdm(range(length)):
                ret, frame = cap.read()     # BGR
                if not ret:
                    break
                ts = index/fps
                if f"{ts:.06f}" not in time_stamps:
                    continue
                count += 1
                
                model     = self.load_model(model_type)
                predictor = SamPredictor(model)
                predictor.set_image(frame, image_format='BGR')

                index_in_ts = time_stamps.index(f"{ts:.06f}")
                input_box   = input_all_boxes[index_in_ts]
                input_point = np.array(input_all_point[index_in_ts])

                if len(input_point) > 0:
                    input_point = input_point[5:]   # exclude head points
                    prob        = input_point[:, 2]
                    
                    # the thresh is important for different model types
                    input_point = input_point[prob>visibel_thresh][:, :2]      
                    input_label = [1] * len(input_point)
                else:
                    input_point = None
                    input_label = None
                
                if len(input_box) <= 0:
                    output_masks.append([])
                    mask_image = frame
                    
                elif type(input_box[0]) == list:
                    input_boxes, new_boxes = multi_boxes(input_box, predictor, frame)

                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=new_boxes,
                        multimask_output=False,
                    )
                    for mask in masks:
                        mask_image = load_mask(mask.cpu().numpy(), random_color)

                    for box in input_boxes:
                        frame = load_box(box.cpu().numpy(), frame)

                    # output_masks.append(compress_masks(masks))

                elif type(input_box[0]) == int or type(input_box[0]) == float:
                    input_boxes = np.array(input_box)[None, :]

                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=input_boxes,
                        multimask_output=multimask_output,
                    )
                    index = np.argmax(scores)
                    select_mask = masks[index]
                    coord_mask = compress_masks(select_mask)
                    if save_video:
                        mask_image = load_mask(select_mask, random_color)
                        (x1, x2, y1, y2) = get_box(coord_mask[:, [1,0]], 
                                    sequence.cam['height'], 
                                    sequence.cam['width'], 
                                    pose_dim=2,
                                    ratio=0)
                        frame = load_box([x1, y1, x2, y2], frame)
                        frame = load_box(input_box, frame, (0, 0, 255))
                    output_masks.append(coord_mask)

                if save_video:
                    mask_image = cv2.add(frame, mask_image)
                    out.write(mask_image)

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_masks
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default='',  help="xxx")
    
    parser.add_argument("--base_path", type=str, default='/wd8t/sloper4d_publish/seq003_street_002', help="xxx")
    
    parser.add_argument("--vid_path", type=str, default=None, help="video path")
    
    parser.add_argument("--pkl_path", type=str, default=None, help="xxx")

    parser.add_argument("--smpl_box", action='store_true', 
                        help="whether to use the SMPL projection as the input box prompt")
    
    parser.add_argument('--thresh', type=float, default=0.6, 
                        help="the keypoints threshold that used to as point prompt")
    
    parser.add_argument('--save_video', action='store_true',
                        help='output a video when segment the video')
    
    parser.add_argument('--over_write', action='store_true',
                        help='whether to overwrite the pkl file for new bbox')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.img_path):
        output_path = f"{args.img_path[:-4]}_out.jpg"
        results = SegManualMaskPredictor().image_predict(
            source           = args.img_path,
            output_path      = output_path,
            model_type       = "vit_l", # vit_l, vit_h, vit_b
            # input_point    = np.array([[100, 100], [200, 200]]),
            input_label      = [0, ],
            input_box        = [903, 443, 950, 549],
            multimask_output = False,
            random_color     = False,
            show             = False,
            save             = True,
        )

    seq_name = os.path.basename(args.base_path)
    pkl_file = os.path.join(args.base_path, f"{seq_name}_labels.pkl") if args.pkl_path is None else args.pkl_path
    vid_path = os.path.join(args.base_path,'rgb_data', f"{seq_name}.MP4") if args.vid_path is None else args.vid_path

    print(pkl_file)
    print(vid_path)

    if os.path.exists(vid_path) and os.path.exists(pkl_file):
        sequence = SLOPER4D_Loader(pkl_file, return_torch=False, return_smpl=True)
        out_fps  = sequence.cam['fps']
        valid_ts = [ts[:-4] for ts in sequence.file_basename]
        if args.smpl_box:
            # load bboxes from the SMPL projection
            boxes = load_boxes(sequence.smpl_mask, sequence.bbox, sequence.cam, ratio=0.1)
            kpts = [[]] * len(boxes)
        else:
            boxes = sequence.bbox
            kpts = sequence.skel_2d

        results = SegManualMaskPredictor().video_predict(
            source           = vid_path,
            model_type       = "vit_l", # vit_l, vit_h, vit_b
            input_all_boxes  = boxes,
            input_all_point  = kpts,
            multimask_output = True,
            random_color     = False,
            output_path      = f"{vid_path[:-4]}_mask.mp4",
            out_fps          = out_fps,
            time_stamps      = valid_ts,
            save_video       = args.save_video,
            visibel_thresh   = args.thresh
        )

        for imgname, mask in zip(sequence.file_basename, results):
            if len(mask) > 200:
                (x1, x2, y1, y2) = get_box(mask[:, [1,0]], 
                            sequence.cam['height'], 
                            sequence.cam['width'], 
                            pose_dim=2,
                            ratio=0)
                sequence.updata_pkl(imgname, bbox=[x1, y1, x2, y2])

        sequence.save_pkl(overwrite=args.overwrite)
        
        with open(pkl_file[:-4] + "_mask.pkl", 'wb') as f:
            pickle.dump({'masks': results}, f)