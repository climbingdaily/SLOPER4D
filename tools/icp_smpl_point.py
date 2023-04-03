
################################################################################
# File: \vis_3d_box.py                                                         #
# Created Date: Sunday July 17th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

import numpy as np
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
import torch
import sys

sys.path.append(".")
sys.path.append("..")
from smpl import SMPL

def vertices_to_root(vertices, index = 0):
    smpl = SMPL()
    return smpl.get_full_joints(torch.FloatTensor(vertices))[..., index, :]

def hidden_point_removal(pcd, camera_location = [0, 0, 0]):
    dist = np.linalg.norm(pcd.get_center())
    radius = dist * 1000
    _, pt_map = pcd.hidden_point_removal(camera_location, radius)
    pcd = pcd.select_by_index(pt_map)
    return pcd

def select_points_on_the_scan_line(points, 
                        view_point=None, 
                        scans=64, 
                        line_num=1024, 
                        fov_up=17.497, 
                        fov_down=-15.869, 
                        precision=1.1):
    """
    > It takes a point cloud and returns a point cloud that has been filtered to only include points
    that are on the scan line
    
    Args:
      points: the point cloud to be processed
      view_point: the point from which the point cloud is viewed. If it is None, the point cloud is
    viewed from the origin.
      scans: the number of vertical bins. Defaults to 64
      line_num: the number of points in a scan line. Defaults to 1024
      fov_up: the upper limit of the vertical field of view of the lidar
      fov_down: the angle of the bottom of the scan line
      precision: the precision of the points, which is the standard deviation of the Gaussian noise
    added to the points.
    
    Returns:
      A point cloud.
    """
    fov_up = np.deg2rad(fov_up)
    fov_down = np.deg2rad(fov_down)
    fov = abs(fov_down) + abs(fov_up)

    ratio = fov/(scans - 1)   # 64bins 的竖直分辨率
    hoz_ratio = 2 * np.pi / (line_num - 1)    # 64bins 的水平分辨率
    # precision * np.random.randn() 
    
    if view_point is not None:
        points -= view_point
    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    
    saved_box = { s:{} for s in np.arange(scans)}

    #### 筛选fov范围内的点
    for idx in range(0, points.shape[0]):
        rule1 =  pitch[idx] > fov_down
        rule2 =  pitch[idx] < fov_up
        rule3 = abs(pitch[idx] % ratio) < ratio * 0.4
        rule4 = abs(yaw[idx] % hoz_ratio) < hoz_ratio * 0.4
        if rule1 and rule2:
            scanid = np.rint((pitch[idx] + 1e-4) / ratio) + scans // 2
            pointid = np.rint((yaw[idx] + 1e-4) // hoz_ratio)
            if scanid < 0 or scanid >= scans:
                continue
            if pointid > 0 and scan_x[idx] < 0:
                pointid += 1024 // 2
            elif pointid < 0 and scan_y[idx] < 0:
                pointid += 1024 // 2
            
            z = np.sin(scanid * ratio + fov_down)
            xy = abs(np.cos(scanid * ratio + fov_down))
            y = xy * np.sin(pointid * hoz_ratio)
            x = xy * np.cos(pointid * hoz_ratio)

            # 找到根指定激光射线夹角最小的点
            cos_delta_theta = np.dot(points[idx], np.array([x, y, z])) / depth[idx]
            delta_theta = np.arccos(abs(cos_delta_theta))
            if pointid in saved_box[scanid]:
                if delta_theta < saved_box[scanid][pointid]['delta_theta']:
                    saved_box[scanid][pointid].update({'points': points[idx], 'delta_theta': delta_theta, 'id': idx})
            else:
                saved_box[scanid][pointid] = {'points': points[idx], 'delta_theta': delta_theta, 'id': idx}

    # save_points  =[]
    id_list = []
    for key, value in saved_box.items():
        if len(value) > 0:
            for k, v in value.items():
                # save_points.append(v['points']) 
                id_list.append(v['id']) 

    # save_points = np.array(save_points)
    # pc=o3d.open3d.geometry.PointCloud()
    # if len(save_points) > 0:
    #     pc.points= o3d.open3d.utility.Vector3dVector(save_points)
    #     pc.paint_uniform_color([0.5, 0.5, 0.5])

    return id_list

def select_visible_points(mesh, rt=np.eye(4), scans=128):
    """
    > Given a mesh, it returns the points that are visible from the mesh
    
    Args:
      mesh: the mesh object
      rt: the rotation and translation of the camera.
    """
    if rt.shape[0] == 7:
        _rt = np.eye(4)
        _rt[:3, 3] = rt[:3]
        _rt[:3, :3] = R.from_quat(rt[3:7]).as_matrix()
        rt = _rt

    mesh_points = (mesh - rt[:3, 3]) @ rt[:3, :3] # (R.T * P).T = P.T * R

    def HPR(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
        dist = np.linalg.norm(pcd.get_center())
        radius = dist * 1000
        _, pt_map = pcd.hidden_point_removal([0,0,0], radius)
        return sorted(pt_map)

    vis_idx = np.array(HPR(mesh_points))

    if scans==64:
        id_list = select_points_on_the_scan_line(mesh_points[vis_idx],  
                            view_point=None, 
                            scans=64, 
                            line_num=1024, 
                            fov_up=17.497, 
                            fov_down=-15.869, 
                            precision=1.1)
    elif scans==128:
        id_list = select_points_on_the_scan_line(mesh_points[vis_idx],  
                            view_point=None, 
                            scans=scans, 
                            line_num=1024, 
                            fov_up=21.11, 
                            fov_down=-21.62, 
                            precision=1.1)
    else:
        print("This LiDAR type is not implemented yet!!")
        exit(-1)

    return sorted(vis_idx[id_list])

def icp_mesh2point(mesh, points, transform, visible_list=None, init_transform=np.eye(4), max_correspondence_distance=0.1):
    """
    > Given a mesh and a set of points, find the translation that minimizes the distance between the
    mesh and the points
    
    Args:
      mesh: the mesh to be aligned
      points: The point cloud to be aligned.
      transform: the camera's rotation and translation in world space.
      init_transform: The initial transformation matrix.
      max_correspondence_distance: The maximum distance between two correspondent points in source and
    target.
    
    Returns:
      The deltaT is the difference between the root position of mesh before and after ICP
    """
    if visible_list is None:
        visible_list = select_visible_points(mesh, transform)
    
    smpl = o3d.geometry.PointCloud()
    smpl.points = o3d.utility.Vector3dVector(mesh)
    smpl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    smpl.select_by_index(visible_list)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    if smpl.has_points() and point_cloud.has_points():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        init_transform = np.eye(4)
        init_transform[:3, 3] = point_cloud.get_center() - smpl.get_center()

        reg_p2l = o3d.pipelines.registration.registration_icp(
            smpl, point_cloud, max_correspondence_distance, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        rt = reg_p2l.transformation
        if reg_p2l.inlier_rmse < 0.1 and reg_p2l.fitness > 0.799:
            root_pos = vertices_to_root(mesh[None, ...])[0].numpy()
            deltaT = rt[:3, :3] @ root_pos + rt[:3, 3] - root_pos
        else:
            deltaT = init_transform[:3, 3]
            deltaT[2] = 0
    else:
        deltaT = np.array([0., 0., 0.])
    
    return deltaT

def icp_mesh_and_point(mesh, 
                      points, 
                      rt=np.eye(4),
                      max_correspondence_distance = 0.2):

    """
    It takes a mesh and a point cloud, and returns the mesh transformed to align with the point cloud
    
    Args:
      mesh: the mesh you want to align
      point_cloud: The point cloud to which the mesh will be aligned.
      rt: the initial pose of the mesh
      max_correspondence_distance: The maximum distance between two correspondent points in source and
    target.
    
    Returns:
      the registered mesh and the transformation matrix.
    """
    if rt.shape[0] == 7:
        _rt = np.eye(4)
        _rt[:3, 3] = rt[:3]
        _rt[:3, :3] = R.from_quat(rt[3:7]).as_matrix()
        rt = _rt
    rrt = np.linalg.inv(rt)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh)
    mesh=m

    maxb = point_cloud.get_axis_aligned_bounding_box().max_bound
    minb = point_cloud.get_axis_aligned_bounding_box().min_bound

    reg_mesh = copy.deepcopy(mesh)

    tt = np.eye(4)
    tt[:3, 3] =  (maxb+minb)/2 - mesh.get_center()
    # reg_mesh.translate((maxb+minb)/2 - mesh.get_center())
    reg_mesh.transform(rrt @ tt)
    point_cloud.transform(rrt)

    smpl = o3d.geometry.PointCloud()
    smpl.points = o3d.utility.Vector3dVector(np.asarray(reg_mesh.vertices))

    smpl = hidden_point_removal(smpl)
    init_transform = np.eye(4)
    id_list = select_points_on_the_scan_line(np.asarray(smpl.points))
    smpl = smpl.select_by_index(id_list)

    if smpl.has_points():
        smpl.estimate_normals()
        point_cloud.estimate_normals()
        reg_p2l = o3d.pipelines.registration.registration_icp(
            smpl, point_cloud, max_correspondence_distance, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # reg_mesh.paint_uniform_color([1, 0.706, 0])
        transformation = reg_p2l.transformation
        reg_mesh.transform(transformation)
    else:
        transformation = np.eye(4)

    reg_mesh.transform(rt)
    point_cloud.transform(rt)

    sumRT = rt @ transformation @ rrt @ tt
    deltaT = vertices_to_root(np.array(reg_mesh.vertices)[None, ...]) - vertices_to_root(np.array(mesh.vertices)[None, ...])
    return deltaT.numpy().squeeze()

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("--mesh", '-B', type=str, default='C:\\Users\\DAI\\Desktop\\temp\\0417-03_tracking.pkl')
    parser.add_argument("--point_cloud", '-M', type=str, default='New Folder')
    args = parser.parse_args() 
    mesh = o3d.io.read_triangle_mesh('C:\\Users\\DAI\\Desktop\\temp\\2_0320.ply')
    pointcoud = o3d.io.read_point_cloud('C:\\Users\\DAI\\Desktop\\temp\\2_0320.pcd')
    trajs = np.loadtxt('C:\\Users\\DAI\\Desktop\\temp\\synced_lidar_trajectory.txt')

    _, deltaT = icp_mesh_and_point(mesh, pointcoud, trajs[trajs[:,0].tolist().index(320)][1:8], max_correspondence_distance =0.1,)
    o3d.io.write_triangle_mesh('C:\\Users\\DAI\\Desktop\\temp\\test_reg_mesh.ply', mesh.translate(deltaT))