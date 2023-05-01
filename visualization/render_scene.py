import os
import sys

# FIXME: better way?
root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(root_folder)
# for pyrender
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # or egl

import time
import tqdm
import argparse
import warnings

import cv2
import torch
import smplx

import numpy as np

from Render import Renderer
from vis_utils import test_opencv_video_format, plot_points_on_img, extrinsic_to_cam, plot_coco_annotation, get_bool_array_from_coordinates, load_mask
from utils import load_point_cloud

from src import  SLOPER4D_Loader

class SenceRender(object):
    def __init__(self, args, model_type='smpl'):
        self.seq_name        = os.path.basename(args.base_path)
        self.rgb_base        = os.path.join(args.base_path, "rgb_data", self.seq_name)
        self.img_base_path   = os.path.join(args.base_path, "rgb_data", f"{self.seq_name}_imgs")
        self.pc_base_path    = os.path.join(args.base_path, "lidar_data", "lidar_frames_rot")
                                            
        self.draw_smpl       = args.draw_smpl
        self.draw_coco17     = args.draw_coco17
        self.draw_human_pc   = args.draw_human_pc
        self.draw_scene_pc   = args.draw_scene_pc
        self.draw_mask       = args.draw_mask

        self.sequence  = SLOPER4D_Loader(os.path.join(args.base_path, self.seq_name+"_labels.pkl"), 
                                         return_torch=False)

        # camera information
        self.cam_fps   = self.sequence.cam['fps']
        self.cam_in    = self.sequence.cam['intrinsics']
        self.cam_dist  = self.sequence.cam['dist']
        self.cam_width  = self.sequence.cam['width']
        self.cam_height = self.sequence.cam['height']

        num_joints = 17
        if args.losstype == 'MSELoss'.lower():
            self.vis_thres = [0.4] * num_joints
        elif 'Regression'.lower() in args.losstype:
            self.vis_thres = [0.05] * num_joints
        elif args.losstype == 'Combined'.lower():
            if num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num 

        # prepare files for visualization
        os.makedirs(self.rgb_base, exist_ok=True)
        if self.draw_coco17:
            if args.index < 0:
                self.coco17_ = self._prepare_output_(self.rgb_base, self.seq_name+'_coco17.mp4')

        if self.draw_mask and args.index < 0:
            self.masks_ = self._prepare_output_(self.rgb_base, self.seq_name+'_mask.mp4')

        if self.draw_smpl:
            self._create_human_model(model_type)
            self.renderer = Renderer(resolution=(self.cam_width, self.cam_height), wireframe=args.wireframe)
            if args.index < 0:
                self.img_smpl_ = self._prepare_output_(self.rgb_base, self.seq_name+'_smpl.mp4')

        if self.draw_human_pc and args.index < 0:
            self.human_pc_ = self._prepare_output_(self.rgb_base, self.seq_name+'_human_pc.mp4')

        if self.draw_scene_pc and args.index < 0:
            self.scene_pc_ = self._prepare_output_(self.rgb_base, self.seq_name+'_scene_pc.mp4')
        
    def _create_human_model(self, model_type='smpl'):
        """call after _load_sence()"""
        self.human_model = smplx.create(args.smpl_model_path, 
                                       model_type=model_type,
                                       gender=self.sequence.smpl_gender, 
                                       use_face_contour=False,
                                       ext="npz")
        
        self.smpl_color = np.array([228/255, 60/255, 60/255]) # colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

    def _prepare_output_(self, save_path, basename, fps=0, w=0, h=0):
        fps = self.cam_fps if fps == 0 else fps
        w   = self.cam_width if w == 0 else w
        h   = self.cam_height if h == 0 else h

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        codec, file_ext = (
            ("mp4v", ".mp4") if test_opencv_video_format("mp4v", ".mp4") else ("x264", ".mkv")
        )
        # if codec == "mp4v":
        #     warnings.warn("x264 codec not available, switching to mp4v")

        if os.path.isdir(save_path):
            output_fname = os.path.join(save_path, basename)
            output_fname = os.path.splitext(output_fname)[0]+ file_ext
        else:
            assert 0, "Please specify a directory with args.output"

        if os.path.isfile(output_fname): 
            bak_file = os.path.join(output_fname+'_%lf'%(time.time())+'.bak')
            os.rename(output_fname, bak_file)

        assert not os.path.isfile(output_fname), output_fname

        return cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=fps,
            frameSize=(w, h),
            isColor=True,
        )

    def run(self, index = -1):
        count = -1
        for sample in tqdm.tqdm(self.sequence):
            count += 1
            if index >= 0 and count != index:
                continue

            img_path = os.path.join(self.img_base_path, sample['file_basename'])
            origin_img = cv2.imread(img_path)
            extrinsic = sample['cam_pose']
            cv2.putText(origin_img, f"{count:05d}: {sample['file_basename']}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
            # smpl
            if self.draw_smpl:
                if len(sample['smpl_pose']) != 0: 
                    with torch.no_grad():
                        smpl_md = self.human_model(betas=sample['betas'].reshape(-1,10), 
                                                return_verts=True, 
                                                global_orient=sample['smpl_pose'][:3].reshape(-1,3),
                                                body_pose=sample['smpl_pose'][3:].reshape(-1, 23*3),
                                                transl=sample['global_trans'].reshape(-1,3))
                    
                    smpl_verts  = smpl_md.vertices.detach().cpu().numpy().squeeze()
                    
                    img = self.renderer.render(
                        origin_img.copy(),
                        smpl_model    = (smpl_verts, self.human_model.faces),
                        cam           = (self.cam_in, extrinsic_to_cam(extrinsic)),
                        color         = self.smpl_color,
                        human_pc      = None,
                        sence_pc_pose = None,
                        mesh_filename = None)
                else:
                    img = origin_img
                    
                if index >= 0:
                    cv2.imwrite(os.path.join(self.rgb_base, f'{self.seq_name}_smpl_{index}.jpg'), img)
                    print(f"image save to: {os.path.join(self.rgb_base, f'{self.seq_name}_smpl_{index}.jpg')}")
                else:
                    self.img_smpl_.write(img)

            # coco17
            if self.draw_coco17 or self.draw_mask:
                img = plot_coco_annotation(origin_img.copy(),
                    keypoints = np.array([sample['skel_2d']]) if self.draw_coco17 else None,
                    bboxes = np.array([sample['bbox'], ]),
                    mask = np.array(sample['mask']) if self.draw_mask else None,
                    _KEYPOINT_THRESHOLD= self.vis_thres)
                    
                if index >= 0:
                    cv2.imwrite(os.path.join(self.rgb_base, f'{self.seq_name}_coco17_{index}.jpg'), img)
                    print(f"image save to: {os.path.join(self.rgb_base, f'{self.seq_name}_coco17_{index}.jpg')}")
                else:
                    self.coco17_.write(img)

            # human pc 
            if self.draw_human_pc:
                if len(sample['human_points']) != 0:
                    img = plot_points_on_img(origin_img.copy(), 
                                             np.array(sample['human_points']), 
                                             extrinsic, 
                                             self.cam_in, 
                                             self.cam_dist)
                else:
                    img = origin_img
                    
                if index >= 0:
                    cv2.imwrite(os.path.join(self.rgb_base, f'{self.seq_name}_human_pc_{index}.jpg'), img)
                    print(f"image save to: {os.path.join(self.rgb_base, f'{self.seq_name}_human_pc_{index}.jpg')}")
                else:
                    self.human_pc_.write(img)

            # scene point cloud
            if self.draw_scene_pc:
                pcd_name = f"{sample['lidar_tstamps']:.03f}".replace('.', '_') + '.pcd'
                pc_path  = os.path.join(self.pc_base_path, pcd_name)
                sence_pc = load_point_cloud(pc_path)
                img      = plot_points_on_img(origin_img.copy(), 
                                              np.asarray(sence_pc.points), 
                                              extrinsic, 
                                              self.cam_in, 
                                              self.cam_dist,
                                              )
                
                if index >= 0:
                    cv2.imwrite(os.path.join(self.rgb_base, f'{self.seq_name}_scene_pc_{index}.jpg'), img)
                    print(f"image save to: {os.path.join(self.rgb_base, f'{self.seq_name}_scene_pc_{index}.jpg')}")
                else:
                    self.scene_pc_.write(img)

            if self.draw_mask:
                coordinate = np.array(sample['mask'])
                masks = get_bool_array_from_coordinates(coordinate)[None, :, :]
                img = load_mask(masks, False)
                # img = load_box(sample['bbox'], origin_img.copy())
                # img = cv2.add(img, mask_image)

                if index >= 0:
                    cv2.imwrite(os.path.join(self.rgb_base, f'{self.seq_name}_mask_{index}.jpg'), img)
                else:
                    self.masks_.write(img)


def parse_args():
    file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pkl_name", type=str, required=True, help="xxx")
    parser.add_argument('base_path', type=str,
                        help='path to sequence folder')
    parser.add_argument('--losstype', type=str, default='regression',
                        help='coco keypoints loss type, used for visualization')
    parser.add_argument('--index', type=int, default=-1,
                        help='the index frame to be saved to a image')
    
    parser.add_argument('--draw_coco17', action='store_true',
                        help='draw COCO17 to images and save as video.')
    parser.add_argument('--draw_mask', action='store_true',
                        help='draw masks to images and save as video.')
    parser.add_argument('--draw_smpl', action='store_true',
                        help='draw SMPL to images and save as video.')
    parser.add_argument('--draw_human_pc', action='store_true',
                        help='draw human point cloud to images and save as video.')
    parser.add_argument('--draw_scene_pc', action='store_true',
                        help='draw scene point cloud to images and save as video.')
    
    parser.add_argument('--smpl_model_path', type=str, default=f"{os.path.dirname(file_path)}/smpl",
                        help='path to SMPL models')
    parser.add_argument('--wireframe', type=bool, default=False,
                        help='render all meshes as wireframes.')
    parser.add_argument('--save_obj', type=bool, default=False,
                        help='save results as .obj files.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    draw_options = {
        'coco': args.draw_coco17,
        'mask': args.draw_mask,
        'smpl': args.draw_smpl,
        'human_pc': args.draw_human_pc,
        'scene_pc': args.draw_scene_pc,
        }
    
    if not np.any(list(draw_options.values())):
        args.draw_coco17 = True
        args.draw_mask = True
        args.draw_smpl = True
        args.draw_human_pc = True
        args.draw_scene_pc = True

    print(f"Renderring sequence in: {args.base_path}")

    det = SenceRender(args)
    det.run(args.index)