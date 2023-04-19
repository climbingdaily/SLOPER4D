import os
import sys

# FIXME: better way?
root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(root_folder)

import time
import tqdm
import argparse
import warnings

import math
import cv2
import torch
import smplx
import trimesh


import pickle
from Render import Renderer
from Visualizer import HumanVisualizer
from utils import *


# for pyrender
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # or egl


class SenceRender(object):
    def __init__(self, args):
        self.draw_coco17 = args.draw_coco17
        self.draw_smpl = args.draw_smpl
        self.draw_human_pc = args.draw_human_pc
        self.draw_scene_pc = args.draw_scene_pc
        self.draw_coco17_kps = args.draw_coco17_kps
        self.save_path = args.base_path
        self.save_path = os.path.join(args.base_path, "rgb_data")
        with open(os.path.join(args.base_path, args.pkl_name+"_labels.pkl"), 'rb') as f:
            self.pkl_data = pickle.load(f)
        self.frame_dev = 1 / self.pkl_data['LiDAR_info']['fps'] / 2
        self.R1 = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        # FIXME
        self.img_base_path = args.img_base_path
        self.scene_pc_base_path = args.scene_pc_base_path
        self._load_cam_info()
        ### FIXME: for test
        vdo_basename = args.pkl_name
        out_base_dir = os.path.join(self.save_path, vdo_basename)
        os.makedirs(out_base_dir, exist_ok=True)
        if self.draw_coco17:
            self.visualizer = HumanVisualizer(args.cfg_file)
            self.coco17_vdo = self._prepare_output_vdo(out_base_dir, vdo_basename+'_coco17.mp4', self.pkl_data['RGB_info']['fps'], self.im_width, self.im_height)
        if self.draw_coco17_kps:
            self.coco17_kps_vdo = self._prepare_output_vdo(out_base_dir, vdo_basename+'_coco17_kps.mp4', self.pkl_data['RGB_info']['fps'], self.im_width, self.im_height)
        if self.draw_smpl:
            self._create_smpl()
            self.renderer = Renderer(resolution=(self.im_width, self.im_height), wireframe=args.wireframe)
            self.img_smpl_vdo = self._prepare_output_vdo(out_base_dir, vdo_basename+'_smpl.mp4', self.pkl_data['RGB_info']['fps'], self.im_width, self.im_height)
        if self.draw_human_pc:
            self.human_pc_vdo = self._prepare_output_vdo(out_base_dir, vdo_basename+'_human_pc.mp4', self.pkl_data['RGB_info']['fps'], self.im_width, self.im_height)
        if self.draw_scene_pc:
            self.scene_pc_vdo = self._prepare_output_vdo(out_base_dir, vdo_basename+'_scene_pc.mp4', self.pkl_data['RGB_info']['fps'], self.im_width, self.im_height)
        
    def _create_smpl(self):
        """call after _load_sence()"""
        self.smpl_model = smplx.create(args.smpl_model_path, model_type="smpl",
                gender=self.pkl_data["SMPL_info"]['gender'], use_face_contour=False,
                num_betas=10, ext="npz")
        self.smpl_color = np.array([147/255, 58/255, 189/255]) # colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

    def _load_cam_info(self):
        self.cam_in = np.array(self.pkl_data['RGB_info']['intrinsics'])
        self.cam_ex = np.array(self.pkl_data['RGB_info']['lidar2cam'])
        self.cam_dist = np.zeros_like(self.pkl_data['RGB_info']['dist'])
        self.im_width, self.im_height = self.pkl_data['RGB_info']['width'], self.pkl_data['RGB_info']['height']

    def _prepare_output_vdo(self, save_path, basename, fps, w, h):
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
        # cam_ex_pre = self.cam_ex
        for idx in tqdm.tqdm(range(self.pkl_data['frame_num'])):
            img_path = os.path.join(self.img_base_path, self.pkl_data['RGB_frames']['file_basename'][idx])
            cam_pose = self.pkl_data['RGB_frames']['cam_pose'][idx]
            cam_ex = np.array(self.pkl_data['RGB_frames']['extrinsic'][idx])
            # smpl
            if self.draw_smpl:
                im = cv2.imread(img_path)
                if len(self.pkl_data['RGB_frames']['smpl_pose']) != 0: 
                    poses = torch.tensor(self.pkl_data['RGB_frames']['smpl_pose'][idx])
                    beta = torch.tensor(self.pkl_data['RGB_frames']['beta'][idx])
                    trans = torch.tensor(self.pkl_data['RGB_frames']['global_trans'][idx])
                    smpl_md = self.smpl_model(betas=beta.reshape(-1,10), return_verts=True, 
                            global_orient=poses[:3].reshape(-1,3),
                            body_pose=poses[3:].reshape(-1, 23*3),
                            transl=trans.reshape(-1,3))
                    smpl_verts = smpl_md.vertices.detach().cpu().numpy().squeeze()
                    smpl_joints = smpl_md.joints.detach().cpu().numpy().squeeze()
                    # # rener SMPL mpdel
                    
                    cam_pose_render = np.eye(4)
                    cam_pose_render[:3, :3] = cam_pose[:3, :3].T @ self.R1[:3,:3]
                    cam_pose_render[:3, -1] = -cam_pose[:3, :3].T @ cam_pose[:3,-1]
                    im = self.renderer.render(
                        im,
                        smpl_model=(smpl_verts, self.smpl_model.faces),
                        cam=(self.cam_in, cam_pose_render),
                        color=self.smpl_color,
                        human_pc = None,
                        sence_pc_pose = None,
                        mesh_filename=None,
                    )
                self.img_smpl_vdo.write(im)
            # coco17
            if self.draw_coco17:
                im = cv2.imread(img_path)
                if len(self.pkl_data['RGB_frames']['bbox'][idx]) != 0: 
                    # print(np.array(img_label["proposals"][-1]['bbox']).shape)
                    im = self.visualizer.draw_predicted_humans(
                        im, [0, ], # id 
                        np.array([self.pkl_data['RGB_frames']['bbox'][idx], ]),
                        np.array([self.pkl_data['RGB_frames']['skel_2d'][idx], ])
                    )
                self.coco17_vdo.write(im)
            # render 2D kps to image
            if self.draw_coco17_kps:
                im = cv2.imread(img_path)
                if len(self.pkl_data['RGB_frames']['bbox'][idx]) != 0: 
                    cocoskel = np.array(self.pkl_data['RGB_frames']['skel_2d'][idx])
                    draw_skel2image(im, cocoskel[:,:-1])
                self.coco17_kps_vdo.write(im)
            # human pc 
            if self.draw_human_pc:
                im = cv2.imread(img_path)
                if len(self.pkl_data['RGB_frames']['human_points'][idx]) != 0:
                    human_pc = np.array(self.pkl_data['RGB_frames']['human_points'][idx])
                    depth = np.sqrt(np.sum(human_pc*human_pc, axis=1))
                    depth = depth / (np.sum(depth)/depth.shape[0]) * 255
                    depth = np.array([
                            155 * np.log2(i/100) / np.log2(864) + 200 if i > 200 else i \
                            for i in depth ])
                    rgb = plt.get_cmap('hsv')(depth/255)[:, :3]
                    human_pc = np.concatenate((human_pc, rgb), axis=1)
                    human_pc[:,:3] = human_pc[:,:3]@cam_pose[:3,:3].T + cam_pose[:3,-1].T
                    human_pixel_pc = camera_to_pixel(human_pc[:,:3], self.cam_in, self.cam_dist)
                    im = draw_pc2image(im, human_pixel_pc, human_pc[:, 3:])
                self.human_pc_vdo.write(im)
            # scene point cloud
            if self.draw_scene_pc:
                pc_path = os.path.join(self.scene_pc_base_path, self.pkl_data['RGB_frames']['lidar_fname'][idx])
                im = cv2.imread(img_path)
                sence_pc = read_pcd_with_color(pc_path) # FIXME: !!!!! x<0 is cutted out !!!!!
                sence_pc[:,:3] = sence_pc[:,:3]@cam_ex[:3,:3].T + cam_ex[:3,-1].T
                sence_pixel_pc = camera_to_pixel(sence_pc[:,:3], self.cam_in, self.cam_dist)
                im = draw_pc2image(im, sence_pixel_pc, sence_pc[:, 3:])
                self.scene_pc_vdo.write(im)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_name", 
        type=str, required=True, help="xxx"
    )
    parser.add_argument('--base_path', 
        type=str, required=True,
        help='path to scene files'
    )
    parser.add_argument('--img_base_path', 
        type=str, required=True,
        help='path to image files'
    )
    parser.add_argument('--scene_pc_base_path', 
        type=str, required=True,
        help='path to scene point cloud files'
    )
    parser.add_argument('--draw_coco17', 
        action='store_true',
        help='draw COCO17 to images and save as video.')
    parser.add_argument('--draw_coco17_kps', 
        action='store_true',
        help='draw COCO17 keypoints to images and save as video.')
    parser.add_argument('--draw_smpl', 
        action='store_true',
        help='draw SMPL to images and save as video.')
    parser.add_argument('--draw_human_pc', 
        action='store_true',
        help='draw human point cloud to images and save as video.')
    parser.add_argument('--draw_scene_pc', 
        action='store_true',
        help='draw scene point cloud to images and save as video.')
    # smpl
    parser.add_argument('--smpl_model_path', 
        type=str, default="./smpl_models",
        help='path to SMPL models')
    parser.add_argument('--wireframe', 
        type=bool, default=False,
        help='render all meshes as wireframes.')
    parser.add_argument('--save_obj', 
        type=bool, default=False,
        help='save results as .obj files.')
    parser.add_argument("--cfg_file", 
        default="libs/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        type=str
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    det = SenceRender(args)
    det.run()

