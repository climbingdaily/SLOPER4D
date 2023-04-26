# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from pyrender.camera import IntrinsicsCamera


class Renderer:
    def __init__(self, resolution=(1920,1080), wireframe=False):
        self.resolution = resolution

        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1]
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, smpl_model, cam, human_pc=None, sence_pc_pose=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):
        # SMPL
        verts, faces  = smpl_model
        # vertex_colors = np.ones([verts.shape[0], 4]) * [color[0], color[1], color[2], 0.8]
        smpl_mesh     = trimesh.Trimesh(verts, faces)
        if mesh_filename is not None:
            smpl_mesh.export(mesh_filename)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            roughnessFactor=0.7,
            baseColorFactor=(color[2], color[1], color[0], 1.0)
        )

        smpl_mesh = pyrender.Mesh.from_trimesh(smpl_mesh, material=material)
        # PointCloud
        if human_pc is not None:
            pc_mesh = pyrender.Mesh.from_points(human_pc)
        if sence_pc_pose is not None:
            sence_pc, lidar_pose = sence_pc_pose
            sphere_poses = []
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1., 0, 0, 0.5]
            for x,y,z in sence_pc[:, :3]:
                sm_pose = np.eye(4)
                sm_pose[:3,-1] = np.array([x, y, z])
                sm_pose = lidar_pose@sm_pose
                sphere_poses.append(sm_pose)
            pc_sence_mesh = pyrender.Mesh.from_trimesh(sm, poses=sphere_poses)
        # Camera
        (cam_intrinsics, cam_pose) = cam
        fx, fy, cx, cy = cam_intrinsics
        camera = IntrinsicsCamera(
            fx=fx, fy=fy, 
            cx=cx, cy=cy,
            zfar=1000
        )

        smpl_node = self.scene.add(smpl_mesh, 'smpl_mesh')
        if human_pc is not None: pc_node = self.scene.add(pc_mesh, 'pc_mesh')
        if sence_pc_pose is not None: 
            sence_pc_node = self.scene.add(pc_sence_mesh, 'sence_pc_node')
        cam_node = self.scene.add(camera, pose=cam_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(smpl_node)
        if human_pc is not None: self.scene.remove_node(pc_node)
        if sence_pc_pose is not None: self.scene.remove_node(sence_pc_node)
        self.scene.remove_node(cam_node)

        return image
