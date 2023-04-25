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

import math
import cv2
import torch
import smplx
import trimesh

import numpy as np
import matplotlib.pyplot as plt

import pickle
from Render import Renderer
from Visualizer import HumanVisualizer
from vis_utils import test_opencv_video_format, draw_skel2image, plot_points_on_img
from utils import load_point_cloud

from src import  SLOPER4D_Loader

class SenceRender(object):
    def __init__(self, args, model_type='smpl'):
        self.draw_coco17        = args.draw_coco17
        self.draw_smpl          = args.draw_smpl
        self.draw_human_pc      = args.draw_human_pc
        self.draw_scene_pc      = args.draw_scene_pc
        self.draw_coco17_kps    = args.draw_coco17_kps
        self.save_path          = os.path.join(args.base_path, "rgb_data")
        self.img_base_path      = args.img_base_path
        self.scene_pc_base_path = args.scene_pc_base_path
        _basename               = args.pkl_name
        out_base_dir            = os.path.join(self.save_path, _basename)

        self.sequence  = SLOPER4D_Loader(os.path.join(args.base_path, _basename+"_labels.pkl"), return_torch=False)

        # camera information
        self.cam_in    = self.sequence.rgb_intrinsics
        self.cam_dist  = self.sequence.rgb_dist
        self.im_width  = self.sequence.rgb_width
        self.im_height = self.sequence.rgb_height

        self.R1 = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])

        # prepare files for visualization
        os.makedirs(out_base_dir, exist_ok=True)
        if self.draw_coco17:
            self.visualizer = HumanVisualizer(args.cfg_file)
            self.coco17_ = self._prepare_output_(out_base_dir, _basename+'_coco17.mp4')

        if self.draw_coco17_kps:
            self.coco17_kps_ = self._prepare_output_(out_base_dir, _basename+'_coco17_kps.mp4')

        if self.draw_smpl:
            self._create_human_model(model_type)
            self.renderer = Renderer(resolution=(self.im_width, self.im_height), wireframe=args.wireframe)
            self.img_smpl_ = self._prepare_output_(out_base_dir, _basename+'_smpl.mp4')

        if self.draw_human_pc:
            self.human_pc_ = self._prepare_output_(out_base_dir, _basename+'_human_pc.mp4')

        if self.draw_scene_pc:
            self.scene_pc_ = self._prepare_output_(out_base_dir, _basename+'_scene_pc.mp4')
        
    def _create_human_model(self, model_type='smpl'):
        """call after _load_sence()"""
        self.human_model = smplx.create(args.smpl_model_path, 
                                       model_type=model_type,
                                       gender=self.sequence.smpl_gender, 
                                       use_face_contour=False,
                                       ext="npz")
        
        self.smpl_color = np.array([147/255, 58/255, 189/255]) # colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

    def _prepare_output_(self, save_path, basename, fps=0, w=0, h=0):
        fps = self.sequence.rgb_fps if fps == 0 else fps
        w   = self.sequence.rgb_width if w == 0 else w
        h   = self.sequence.rgb_height if h == 0 else h

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")

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
            fps=int(fps), # FIXME: 
            frameSize=(w, h),
            isColor=True,
        )

    def run(self):
        for sample in tqdm.tqdm(self.sequence):
            img_path = os.path.join(self.img_base_path, sample['file_basename'])
            cam_pose = sample['cam_pose']
            
            # smpl
            if self.draw_smpl:
                img = cv2.imread(img_path)
                if len(sample['smpl_pose']) != 0: 
                    with torch.no_grad():
                        smpl_md = self.human_model(betas=sample['betas'].reshape(-1,10), 
                                                return_verts=True, 
                                                global_orient=sample['smpl_pose'][:3].reshape(-1,3),
                                                body_pose=sample['smpl_pose'][3:].reshape(-1, 23*3),
                                                transl=sample['global_trans'].reshape(-1,3))
                    
                    smpl_verts  = smpl_md.vertices.detach().cpu().numpy().squeeze()
                    # smpl_joints = smpl_md.joints.detach().cpu().numpy().squeeze()
                    
                    view_matrix = np.eye(4)
                    view_matrix[:3, :3] = cam_pose[:3, :3].T @ self.R1[:3,:3]
                    view_matrix[:3, -1] = -cam_pose[:3, :3].T @ cam_pose[:3,-1]

                    img = self.renderer.render(
                        img,
                        smpl_model    = (smpl_verts, self.human_model.faces),
                        cam           = (self.cam_in, view_matrix),
                        color         = self.smpl_color,
                        human_pc      = None,
                        sence_pc_pose = None,
                        mesh_filename = None)
                    
                self.img_smpl_.write(img)

            # coco17
            if self.draw_coco17:
                img = cv2.imread(img_path)
                if len(sample['bbox']) != 0: 
                    img = self.visualizer.draw_predicted_humans(
                        img, 
                        [0, ], # id 
                        np.array([sample['bbox'], ]),
                        np.array([sample['skel_2d'], ]))
                self.coco17_.write(img)

            # render 2D kps to image
            if self.draw_coco17_kps:
                img = cv2.imread(img_path)
                if len(sample['bbox']) != 0: 
                    cocoskel = np.array(sample['skel_2d'])
                    img = draw_skel2image(img, cocoskel[:, :-1])
                self.coco17_kps_.write(img)

            # human pc 
            if self.draw_human_pc:
                if len(sample['human_points']) != 0:
                    human_pc = np.array(sample['human_points'])
                    img = plot_points_on_img(img_path, human_pc, cam_pose, self.cam_in, self.cam_dist)
                else:
                    img = cv2.imread(img_path)
                self.human_pc_.write(img)

            # scene point cloud
            if self.draw_scene_pc:
                img      = cv2.imread(img_path)
                pcd_name = f"{sample['lidar_tstamps']:.03f}".replace('.', '_') + '.pcd'
                pc_path  = os.path.join(self.scene_pc_base_path, pcd_name)
                sence_pc = load_point_cloud(pc_path)
                img      = plot_points_on_img(img_path, 
                                              np.asarray(sence_pc.points), 
                                              cam_pose, 
                                              self.cam_in, 
                                            #   self.cam_dist,
                                              )

                self.scene_pc_.write(img)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_name", type=str, required=True, help="xxx")
    parser.add_argument('--base_path', type=str, required=True,
                        help='path to scene files')
    parser.add_argument('--img_base_path', type=str, required=True,
                        help='path to image files')
    parser.add_argument('--scene_pc_base_path', type=str, required=True,
                        help='path to scene point cloud files')
    parser.add_argument('--draw_coco17', action='store_true',
                        help='draw COCO17 to images and save as video.')
    parser.add_argument('--draw_coco17_kps', action='store_true',
                        help='draw COCO17 keypoints to images and save as video.')
    parser.add_argument('--draw_smpl', action='store_true',
                        help='draw SMPL to images and save as video.')
    parser.add_argument('--draw_human_pc', action='store_true',
                        help='draw human point cloud to images and save as video.')
    parser.add_argument('--draw_scene_pc', action='store_true',
                        help='draw scene point cloud to images and save as video.')
    parser.add_argument('--smpl_model_path', type=str, default="../smpl",
                        help='path to SMPL models')
    parser.add_argument('--wireframe', type=bool, default=False,
                        help='render all meshes as wireframes.')
    parser.add_argument('--save_obj', type=bool, default=False,
                        help='save results as .obj files.')
    parser.add_argument("--cfg_file", type=str,
                        default="libs/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    det = SenceRender(args)
    det.run()

