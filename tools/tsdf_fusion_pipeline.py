################################################################################
# File: /tsdffusion_pipeline.py                                                #
# Created Date: Monday October 31st 2022                                       #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY: borrowed from 
# 2023-03-08	ABC	
# https://github.com/PRBonn/vdbfusion/blob/1d2547526a3c2c088ec12fe614fc9902c154b6a3/examples/python/vdbfusion_pipeline.py#L10                                                                     #
################################################################################

from functools import reduce
import os
import sys
import time

import numpy as np
import open3d as o3d
from tqdm import trange
from glob import glob

from vdbfusion import VDBVolume
from .tool_func import load_point_cloud

def csf_filter(xyz, cloth_resolution = 0.15, 
            rigidness = 3, time_step = 0.65, 
            class_threshold = 0.1, interations = 500):
    import CSF #https://github.com/jianboqi/CSF
    """
    `csf_filter` takes in a point cloud and returns two lists of indices: one for the ground points and
    one for the non-ground points
    
    Args:
    xyz: the point cloud to be filtered.
    cloth_resolution: The resolution of the cloth mesh. The smaller the value, the more accurate the
    simulation, but the more time it takes.
    rigidness: The higher the value, the more rigid the cloth is. Defaults to 3
    time_step: The time step of the simulation.
    class_threshold: The distance threshold between the point cloud and the cloth simulation point.
    interations: The maximum number of iterations to run the algorithm for. Defaults to 500
    
    Returns:
    ground and non_ground are the indices of the points that are classified as ground and non-ground
    respectively.
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False  # 粒子设置为不可移动
    csf.params.cloth_resolution = cloth_resolution  # 布料网格分辨率
    csf.params.rigidness = rigidness  # 布料刚性参数
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold  # 点云与布料模拟点的距离阈值
    csf.params.interations = interations  # 最大迭代次数
    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # 地面点索引列表
    non_ground = CSF.VecInt() # 非地面点索引列表
    csf.do_filtering(ground, non_ground) # 执行滤波
    return ground, non_ground

def remove_small_clusters(mesh, min_face_count):
    """
    Remove small clusters of triangles from a mesh.

    Args:
        mesh: An Open3D triangle mesh.
        min_face_count: An integer specifying the minimum number of triangles in a cluster to keep.

    Returns:
        None
    """
    
    print(f'Clusterring...')
    triangle_clusters, cluster, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster = np.asarray(cluster)
    triangles_to_remove = cluster[triangle_clusters] <= min_face_count
    mesh.remove_triangles_by_mask(triangles_to_remove)

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    print(f'Removed {len(cluster[cluster <= min_face_count])}/{len(cluster)} clusters ({triangles_to_remove.sum()} triangles) with <={min_face_count}  triangles.')

def filter_mesh(mesh, min_face_count:int=40):
    """
    It removes outliers, duplicated triangles, vertices, and degenerate triangles
    
    Args:
      mesh: the mesh to be filtered
      voxel_size (float): The size of the voxel to use for the voxel grid filter.
    
    Returns:
      A mesh object
    """
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    pcd, valid_list = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    mesh = mesh.select_by_index(valid_list)
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_unreferenced_vertices()
    # mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_degenerate_triangles()

    remove_small_clusters(mesh, min_face_count)
    mesh = mesh.remove_unreferenced_vertices()

    return mesh

class VDBFusionPipeline:
    """Abstract class that defines a Pipeline, derived classes must implement the dataset and config
    properties.
    
    > This class takes in a directory of point clouds, a list of poses, and a few other parameters,
    and creates a TSDF volume
    
    Args:
        points_dir (str): the directory where the point cloud data is stored
        poses (list): list of poses for each point cloud
        map_name (str): the name of the map you want to save. Defaults to tsdf
        voxel_size (float): The size of the voxels in the voxel grid.
        sdf_trunc (float): The truncation distance for the TSDF.
        space_carving (str): If True, the TSDF volume will be carved out of the space. This is useful for
    scenes with large empty spaces. Defaults to False
        jump (int): how many scans to skip between each scan. Defaults to 0
        n_scans (int): number of scans to use for the map
    """
    def __init__(self, 
                 points_dir: str, 
                 poses : list, 
                 start : int=0, 
                 end : int=np.inf, 
                 map_name: str='tsdf', 
                 voxel_size: float=0.1, 
                 sdf_trunc: float=0.1, 
                 space_carving: str=False, 
                 jump: int = 0, 
                 n_scans: int = -1):
        
        # load pcd data
        points_list = os.listdir(points_dir)
        points_list = glob(points_dir+'/*.pcd')
        
        def get_file_time(x):
            return float(os.path.split(x)[1].split('.')[0].replace('_', '.'))
        
        points_list = sorted(points_list, key=lambda x: get_file_time(x))[start: end]
        
        assert len(points_list) == len(poses)
        
        self._dataset = [[pt, rt] for pt, rt in zip(points_list, poses)]
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.space_carving = space_carving
        
        self._n_scans = len(self._dataset) if n_scans == -1 else n_scans
        self._jump = jump
        
        self.out_dir = os.path.dirname(points_dir)
        self._map_name = f"{map_name}_{self._n_scans}frames"
        
        self._tsdf_volume = VDBVolume(
            self.voxel_size,
            self.sdf_trunc,
            self.space_carving,
        )
        self._res = {}

    def run(self, skip:int=10, save_vdb:bool=False):
        self._run_tsdf_pipeline(skip=skip)
        self._write_ply()
        # self._write_cfg()
        if save_vdb:
            self._write_vdb()
        self._print_tim()
        self._print_metrics()

    def segment_ground(self):
        ground, non_ground = csf_filter(np.asarray(self._res['mesh'].vertices), 
                                        cloth_resolution=0.1, 
                                        rigidness=1,
                                        class_threshold=0.3)
        ground_map = self._res['mesh'].select_by_index(ground)
        ground_path = os.path.join(self.out_dir, f'{self._map_name}_ground.ply')
        o3d.io.write_triangle_mesh(ground_path, ground_map)
        print(f'TSDF ground map saved in {ground_path}')

    def visualize(self):
        # o3d.visualization.draw_geometries([self._res["mesh"]])
        pass

    def __len__(self):
        return len(self._dataset)

    def _run_tsdf_pipeline(self, skip=10):
        times = []
        for idx in trange(self._jump, self._jump + self._n_scans, skip, unit=" frames", desc='TSDF Fusion'):
            scan, pose = self._dataset[idx]
            tic = time.perf_counter_ns()
            
            if pose.shape[0] == 12:
                pose = pose.reshape(3, 4)
                pose = np.vstack([pose, np.array([0,0,0,1])])
            try:
                pt = load_point_cloud(os.path.join(os.path.dirname(scan) + '_cropped', os.path.basename(scan)), position=pose[:3, 3])
                
            except Exception as e:
                pt = load_point_cloud(scan, position=pose[:3, 3])
            self._tsdf_volume.integrate(np.array(pt.points), pose)
            
            toc = time.perf_counter_ns()
            times.append(toc - tic)
        self._res = {"mesh": self._get_o3d_mesh(self._tsdf_volume), "times": times}

    def _write_vdb(self):
        os.makedirs(self.out_dir, exist_ok=True)
        filename = os.path.join(self.out_dir, self._map_name) + ".vdb"
        self._tsdf_volume.extract_vdb_grids(filename)

    def _write_ply(self):
        os.makedirs(self.out_dir, exist_ok=True)
        filename = os.path.join(self.out_dir, self._map_name) + ".ply"
        o3d.io.write_triangle_mesh(filename, self._res["mesh"])
        print(f'TSDF map saved in {filename}')

    def _write_cfg(self):
        os.makedirs(self.out_dir, exist_ok=True)
        filename = os.path.join(self.out_dir, self._map_name) + ".yml"

    def _print_tim(self):
        total_time_ns = reduce(lambda a, b: a + b, self._res["times"])
        total_time = total_time_ns * 1e-9
        total_scans = self._n_scans - self._jump
        self.fps = float(total_scans / total_time)

    @staticmethod
    def _get_o3d_mesh(tsdf_volume, fill_holes=True, min_weight=0):
        vertices, triangles = tsdf_volume.extract_triangle_mesh(fill_holes, min_weight)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        mesh = filter_mesh(mesh)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5,0.5,0.5])
        return mesh

    def _print_metrics(self):
        # compute size of .vdb file
        filename = os.path.join(self.out_dir, self._map_name) + ".ply"
        file_size = float(os.stat(filename).st_size) / (1024 * 1024)

        # print metrics
        trunc_voxels = int(np.ceil(self.sdf_trunc / self.voxel_size))

        # filename = os.path.join(self.out_dir, self._map_name) + ".txt"
        # with open(filename, "w") as f:
        #     stdout = sys.stdout
        #     sys.stdout = f  # Change the standard output to the file we created.
        print(f"--------------------------------------------------")
        print(f"Results for dataset {self._map_name}:")
        print(f"--------------------------------------------------")
        print(f"voxel size       = {self.voxel_size} [m]")
        print(f"truncation       = {trunc_voxels} [voxels]")
        print(f"space carving    = {self.space_carving}")
        print(f"Avg FPS          = {self.fps:.2f} [Hz]")
        # print(f"--------------------------------------------------")
                # If PYOPENVDB_SUPPORT has not been enabled then we can't report any metrics
        if self._tsdf_volume.pyopenvdb_support_enabled:
            # print("No metrics available, please compile with PYOPENVDB_SUPPORT")
            # Compute the dimensions of the volume mapped
            grid = self._tsdf_volume.tsdf
            bbox = grid.evalActiveVoxelBoundingBox()
            dim = np.abs(np.asarray(bbox[1]) - np.asarray(bbox[0]))
            volume_extent = np.ceil(self.voxel_size * dim).astype(np.int32)
            volume_extent = f"{volume_extent[0]} x {volume_extent[1]} x {volume_extent[2]}"

            # Compute memory footprint
            total_voxels = int(np.prod(dim))
            float_size = 4
            # Always 2 grids
            mem_footprint = 2 * grid.memUsage() / (1024 * 1024)
            dense_equivalent = 2 * (float_size * total_voxels) / (1024 * 1024 * 1024)  # GB
            print(f"volume extent    = {volume_extent} [m x m x m]")
            print(f"memory footprint = {mem_footprint:.2f} [MB]")
            print(f"dense equivalent = {dense_equivalent:.2f} [GB]")
            
        print(f"size on disk     = {file_size:.2f} [MB]")
        print(f"--------------------------------------------------")
        print(f"number of scans  = {len(self)}")
        # print(f"points per scan  = {len(self._dataset[0][0])}")
        # print(f"min range        = {self._config.min_range} [m]")
        # print(f"max range        = {self._config.max_range} [m]")
        print(f"--------------------------------------------------")
            # sys.stdout = stdout

        # Print it
        # os.system(f"cat {filename}")