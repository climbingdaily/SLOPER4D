

bl_info = {"name": "Sequence SMPL viewer",
           "description": "Display sequential SMPL meshes data.",
           "author": "Yudi Dai",
           "version": (1, 0, 0),
           "blender": (3, 2, 0),
           "location": "View3D > Sidebar > Sequence SMPL viewer",
           "warning": "",
           "category": "3D View", }

import os
import bpy
import bmesh

import math
import traceback
import pickle
import numpy as np
import mathutils
from mathutils import Vector
from collections.abc import Iterable

sloper4d_data = None
CAMERA_NAME = 'first_person_view'

MAT_NAMES = {"points": "cbrewer medium blue",
             "first_person":{
                "opt_pose": "cbrewer medium blue",
                "imu_pose": "cbrewer soft blue",
                "baseline2": "cbrewer soft green",
                },
             "second_person":{
                "opt_pose": "cbrewer soft red",
                "imu_pose": "cbrewer medium yellow",
                "baseline2": "cbrewer medium green",
                }
             }

HUMAN_POSE_LIST = [
    'first_person_opt_pose',
    'first_person_baseline2',
    'first_person_imu_pose',
    'second_person_opt_pose',
    'second_person_baseline2',
    'second_person_imu_pose',
    ]

SMPL_VERTS_NUM = 6890

class Sloper4DViewerPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_sloper4d_viewer_panel"
    bl_label = "Sequence SMPL viewer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Sequence SMPL viewer"

    def draw(self, context):
        global sloper4d_data
        layout = self.layout

        # 文件选择框
        row = layout.row()
        row.prop(context.scene, "sloper4d_file_path", text="Pkl Path", icon="FILE")

        # 显示字段信息的垂直布局
        box = layout.box()
        col = box.column()
        
        # 遍历字段信息并显示
        if sloper4d_data:
            for field, values in sloper4d_data.items():
                if isinstance(values, dict):
                    col.label(text=f"{field}:")
                    sub_col = col.column()
                    for sub_field, sub_values in values.items():
                        sub_col.label(text=f"  - {sub_field}: {len(sub_values)} frames")
                elif isinstance(values, Iterable):
                    col.label(text=f"{field}: {len(values)}")
                else:
                    col.label(text=f"{field}: {values}")
        else:
            col.label(text="N/A")

        # 垂直布局用于包裹按钮
#        box = layout.box()

        # Load Data和Erase Data按钮
        r = box.row(align=True)
        r.scale_y = 1.5

        c = r.column()
        c.operator("sloper4d_viewer.load_data", text="Load Data")
        c.enabled = sloper4d_data is None

        c = r.column()
        c.operator("sloper4d_viewer.erase_data", text="Erase Data")
        c.enabled = not sloper4d_data is None
           
        sub = layout.column()

        r = sub.row()
        r.label(text="Set visible SMPL")
        # Button：设置可见的SMPL对象
        r = sub.row(align=True)
        c = r.column()
        c.operator("sloper4d_viewer.set_opt_visible_smpl", text="Opt. Pose")
        c = r.column()
        c.operator("sloper4d_viewer.set_imu_visible_smpl", text="IMU pose")
        c = r.column()
        c.operator("sloper4d_viewer.set_bl2_visible_smpl", text="Baseline2")
        sub.separator()
        
        # 按键：执行函数
        r = sub.row()
        r.prop(context.scene, "execute_function", text="Update as frame change")
        sub.separator()

        r = sub.row()
        r.label(text="Manually update data")
        r = sub.row()
        split = r.split(factor=0.25, align=True)
        c = split.column()
        c.operator("sloper4d_viewer.set_data", text="Set")
        c = split.column()
        c.prop(context.scene, "sloper4d_frame_number", text="Frame No.")
        r = sub.row()
        r.prop(context.scene, "point_cube_size", text="Point Cube Size")
        if context.scene.point_cube_size < 0:
           context.scene.point_cube_size = 0
        elif context.scene.point_cube_size > 50:
           context.scene.point_cube_size = 50
        sub.separator()


        r = sub.row()
        r.label(text="Manually render")
        r = sub.row()
        r.prop(context.scene, "my_output_folder")
        r = sub.row(align=True)
        r.prop(context.scene, "my_start_frame")
        r.prop(context.scene, "my_end_frame")
        r = sub.row()
        r.operator("sloper4d_viewer.my_render", text="Render")

@bpy.app.handlers.persistent
def execute_on_frame_change(scene):
    # Function to execute when the button is pressed
    if scene.execute_function:
        global sloper4d_data
        cur_frame_number = bpy.context.scene.frame_current
        scene.sloper4d_frame_number = cur_frame_number
        cube_size = scene.point_cube_size / 100 # cm --> m
        update_camera(sloper4d_data, cur_frame_number)
        set_data(sloper4d_data, cur_frame_number, cube_size=cube_size)
        
        if bpy.context.area is not None:
            bpy.context.area.tag_redraw()
        # print(f"Executing the function on frame {cur_frame_number}")

class MyRenderOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.my_render"
    bl_label = "Render Frames"

    def execute(self, context):
        start_frame = context.scene.my_start_frame
        end_frame = context.scene.my_end_frame
        output_folder = context.scene.my_output_folder
        
        # # 遍历每一帧，调用 setdata 函数，并渲染图像
        for frame in range(start_frame, end_frame + 1):
            context.scene.frame_set(frame)
            bpy.ops.render.render(write_still=True)
            file_path = os.path.join(bpy.path.abspath(output_folder), f"{frame:04d}.png")
            bpy.data.images['Render Result'].save_render(filepath=file_path)
            print(f"IMG saved to: {file_path}")

        return {'FINISHED'}

class SetOptVisibleSMPLOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.set_opt_visible_smpl"
    bl_label = "Set opt Visible Smpl"

    def execute(self, context):        
        visible_objects = ["first_person_opt_pose", "second_person_opt_pose"]
        set_visible_smpl(visible_objects)
        bpy.context.area.tag_redraw()
        return {"FINISHED"}
    
class SetIMUVisibleSMPLOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.set_imu_visible_smpl"
    bl_label = "Set IMU Visible Smpl"

    def execute(self, context):        
        visible_objects = ["first_person_imu_pose", "second_person_imu_pose"]
        set_visible_smpl(visible_objects)
        bpy.context.area.tag_redraw()
        return {"FINISHED"}
    
    
class SetBaseline2VisibleSMPLOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.set_bl2_visible_smpl"
    bl_label = "Set BL2 Visible Smpl"

    def execute(self, context):        
        visible_objects = ["first_person_baseline2", "second_person_baseline2"]
        set_visible_smpl(visible_objects)
        bpy.context.area.tag_redraw()
        return {"FINISHED"}

def set_visible_smpl(visible_objects):
    # 遍历场景中的所有对象
    for obj in bpy.context.scene.objects:
        # 如果是网格对象并且不在隐藏列表中
        if obj.type == 'MESH' and obj.name in HUMAN_POSE_LIST:
            if obj.name in visible_objects:
                obj.hide_render = False  # 设置为可渲染
                obj.hide_set(False)  # 设置为可见
            else:
                obj.hide_render = True  
                obj.hide_set(True)

class Sloper4DViewerEraseDataOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.erase_data"
    bl_label = "Erase Data"

    def execute(self, context):
        global sloper4d_data

        # 清除sloper4d_data全局变量
        # if 'sloper4d_data' in bpy.context.scene:
        #     del bpy.context.scene['sloper4d_data']
        # bpy.context.area.tag_redraw()
        sloper4d_data = None
        
        return {"FINISHED"}

class Sloper4DViewerLoadDataOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.load_data"
    bl_label = "Load Data"

    def execute(self, context):
        global sloper4d_data
        scene = context.scene
        file_path = bpy.path.abspath(scene.sloper4d_file_path)

        sloper4d_data = self.read_pkl(file_path)
        
        obj = extrinsic_to_blender_camera(np.eye(4), camera_name=CAMERA_NAME)
        
        if sloper4d_data is not None and 'lidar_extrinsics' in sloper4d_data:
            exs = sloper4d_data['lidar_extrinsics']
            frames = np.arange(len(exs))
            scene.execute_function = False
            cur_frame_number = bpy.context.scene.frame_current
            set_camera_keyframes(obj, exs, frames)
            scene.execute_function = True
            scene.frame_set(cur_frame_number)
        
        bpy.context.area.tag_redraw()
        
        return {"FINISHED"}

    def read_pkl(self, file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        if "framerate" in data:
            data["framerate"] = int(data["framerate"])
        data["faces"] = data["faces"].tolist()
            
        return data
    
def set_camera_keyframes(camera_object, extrinsics, frames):
    """
    The function sets keyframes for a camera object's location and rotation based on a given set of
    extrinsics and frames.
    
    Args:
      camera_object: The camera_object parameter is the camera object in Blender that you want to set
    keyframes for.
      extrinsics: The "extrinsics" parameter is a list of camera extrinsic matrices. Each extrinsic
    matrix represents the position and orientation of the camera in the world coordinate system at a
    specific frame.
      frames: The `frames` parameter is a list of frame numbers where you want to set the camera
    keyframes. Each frame number represents a specific point in time in the animation.
    """
    for frame, extrinsic in zip(frames, extrinsics):
        # 设置关键帧所在的帧数
#        camera_matrix = extrinsic_to_cam(extrinsic)
        camera_matrix = extrinsic

        # It will trigger the function 'execute_on_frame_change'
        bpy.context.scene.frame_set(frame)
        
        # 设置相机对象的世界矩阵
        camera_object.matrix_world = mathutils.Matrix(camera_matrix)
        
        # 为相机对象的位置和旋转设置关键帧
        camera_object.keyframe_insert(data_path="location", index=-1)
        camera_object.keyframe_insert(data_path="rotation_euler", index=-1)


class Sloper4DViewerSetDataOperator(bpy.types.Operator):
    bl_idname = "sloper4d_viewer.set_data"
    bl_label = "Set Data"

    def execute(self, context):
        global sloper4d_data
        scene = context.scene
        frame_number = scene.sloper4d_frame_number
        frame_number = 0 if frame_number < 0 else frame_number
        cube_size = scene.point_cube_size / 100 # cm --> m
        
        if bpy.context.scene.execute_function:
            bpy.context.scene.execute_function = False
            bpy.context.scene.frame_set(frame_number)
            bpy.context.scene.execute_function = True
        else:
            bpy.context.scene.frame_set(frame_number)
        
        # 调用set_data()函数
        set_data(sloper4d_data, frame_number, cube_size=cube_size)
        update_camera(sloper4d_data, frame_number)

        bpy.context.area.tag_redraw()
        return {"FINISHED"}

def set_data(data, start, end = -1, cube_size=0.015):
    start = 0 if start < 0 else start
    if end <= start or end <=0:
        end = start + 1

    def add_person_smpl(person):
        for mesh_name, vertices in data[person].items():
            obj = get_or_create_object_by_name(f'{person}_{mesh_name}', data['faces'])
        
            # bpy.context.scene.frame_start = 0
            # bpy.context.scene.frame_end = total_frames - 1
            
            animate_smpl_object(obj, len(vertices), vertices, start, end)
            
            set_material(obj, MAT_NAMES[person][mesh_name])
    
    add_person_smpl('first_person')
    add_person_smpl('second_person')
    generate_cubes(data['point_clouds'][start], cube_size, start)
        
    # print("Done!")


def update_camera(data, frame):
    if data is not None and 'lidar_extrinsics' in data:
        if frame > len(data['lidar_extrinsics']):
            return
        obj = make_camera_mesh_model(d=0.4, fov=70) 
#            cam = extrinsic_to_cam(data['lidar_extrinsics'][frame])
        ex = data['lidar_extrinsics'][frame]
        transform_obj(obj, ex[:3, :3], ex[:3, 3])

def set_material(obj, mat_name):
    if mat_name in bpy.data.materials:
        if obj.data.materials:
            return  # 如果已经设置材质，直接返回，不进行操作
        
        # 获取材质
        material = bpy.data.materials[mat_name]
        
        if 'cbrewer soft red' in mat_name:
            material.node_tree.nodes["Hue Saturation Value"].inputs[4].default_value = (0.9843, 0.45, 0.4471, 1)

        # 为OBJ对象分配材质
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.active_material_index = 0


def generate_cubes(points, size, frame, collection_name='point_clouds'):
    def create_cube_vertices(center):
        half_size = size / 2
        vertices = [
            (center[0] - half_size, center[1] - half_size, center[2] - half_size),
            (center[0] - half_size, center[1] - half_size, center[2] + half_size),
            (center[0] - half_size, center[1] + half_size, center[2] - half_size),
            (center[0] - half_size, center[1] + half_size, center[2] + half_size),
            (center[0] + half_size, center[1] - half_size, center[2] - half_size),
            (center[0] + half_size, center[1] - half_size, center[2] + half_size),
            (center[0] + half_size, center[1] + half_size, center[2] - half_size),
            (center[0] + half_size, center[1] + half_size, center[2] + half_size)
        ]
        return vertices

    def create_cube_faces(offset):
        faces = [
            (0 + offset, 2 + offset, 1 + offset), (1 + offset, 2 + offset, 3 + offset),
            (4 + offset, 5 + offset, 6 + offset), (5 + offset, 7 + offset, 6 + offset),
            (0 + offset, 1 + offset, 4 + offset), (1 + offset, 5 + offset, 4 + offset),
            (2 + offset, 6 + offset, 3 + offset), (3 + offset, 6 + offset, 7 + offset),
            (0 + offset, 4 + offset, 2 + offset), (2 + offset, 4 + offset, 6 + offset),
            (1 + offset, 3 + offset, 5 + offset), (3 + offset, 7 + offset, 5 + offset)
        ]
        return faces

    vertices = []
    faces = []

    for point in points:
        cube_vertices = create_cube_vertices(point)
        cube_faces = create_cube_faces(len(vertices))
        vertices.extend(cube_vertices)
        faces.extend(cube_faces)

    mesh = bpy.data.meshes.new("CubesMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    collection = bpy.data.collections.get(collection_name)
    if not collection:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    object_name = f"p"
    if object_name in bpy.data.objects:
        obj = bpy.data.objects[object_name]
        # 删除物体数据
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    for obj in collection.objects:
        # 设置对象的可见性和渲染属性
        if obj:
            obj.hide_set(True)
            obj.hide_render = True

    obj = bpy.data.objects.new(object_name, mesh)
    set_material(obj, MAT_NAMES['points'])
        
    collection.objects.link(obj)
    

def animate_smpl_object(obj, frames, new_vertices, start=0, end=-1, key_skip=1):
    mesh_data = obj.data
    vertices = mesh_data.vertices
    # scene = bpy.context.scene
    end = frames if end <= 0 else end
    
    for frame in range(start, end):
        # scene.frame_set(frame)
        
        for vertex, new_vert in zip(vertices, new_vertices[frame]):
            # update current coordinates
            try:
                vertex.co = Vector((new_vert[0], new_vert[1], new_vert[2]))
            except:
                traceback.print_exc()
#            if (frame ) % key_skip == 0:
#                vertex.keyframe_insert(data_path="co", index=-1)
        mesh_data.update()

            
def get_or_create_object_by_name(object_name, faces):
    scene = bpy.context.scene
    
    # 检查物体是否存在data but not in scene collection
        
    if object_name in scene.objects:
        obj = bpy.data.objects[object_name] 
    else:
        if object_name in bpy.data.objects:
            obj = bpy.data.objects[object_name] 
            # 删除物体数据
            bpy.data.objects.remove(obj, do_unlink=True)
        # 创建新物体
        mesh = bpy.data.meshes.new(object_name + "_Mesh")
        obj = bpy.data.objects.new(object_name, mesh)
    
        # 将物体添加到场景中
        scene.collection.objects.link(obj)
    
        temp_vertices = np.array([[0.0,0.0,0.0]]*SMPL_VERTS_NUM)

        # 创建网格数据并应用于物体
        mesh_data = obj.data
        print(len(faces))
        mesh_data.from_pydata(temp_vertices, [], faces)
        mesh_data.update()
    
    if obj is not None:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
    
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
    
        bpy.context.view_layer.update()
    
    return obj


def make_camera_mesh_model(d=0.5, fov=70, vfov=45, object_name='LiDAR_View_model'):
    # 计算相机模型的尺寸
    w = d * math.tan(math.radians(fov/2))
    h = d * math.tan(math.radians(vfov/2))

    # 创建相机模型的顶点坐标
    vertices = [
        (0, 0, 0),  # 顶点 origin
        (w, h, -d),  # 顶点 a
        (w, -h, -d),  # 顶点 b
        (-w, -h, -d),  # 顶点 c
        (-w, h, -d),  # 顶点 d
        (0, 0, -d),
    ]

    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 1),
        (1, 2, 5),
        (2, 3, 5),
        (3, 4, 5),
        (4, 1, 5),
    ]
    
    if object_name in bpy.context.scene.objects:
        obj = bpy.data.objects[object_name] 
    else:
        if object_name in bpy.data.objects:
            obj = bpy.data.objects[object_name] 
            # 删除物体数据
            bpy.data.objects.remove(obj, do_unlink=True)

        # 创建相机模型的 Mesh 对象
        mesh = bpy.data.meshes.new(object_name)
        obj = bpy.data.objects.new(object_name, mesh)

        # 将相机模型添加到场景中
        bpy.context.collection.objects.link(obj)

        # 设置相机模型的顶点和边
        mesh.from_pydata(vertices, [], faces)

        # 设置相机模型的材质
        material = bpy.data.materials.new(name="CameraMaterial")
        obj.data.materials.append(material)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        principled_bsdf = nodes.get("Principled BSDF")
        principled_bsdf.inputs["Base Color"].default_value = (0, 0.8, 0, 1)  # 设置颜色为80%绿色
        principled_bsdf.inputs["Alpha"].default_value = 0.4  # 设置透明度为0.3
    
        render_wireframe(obj)
    # 刷新场景
    bpy.context.view_layer.update()

    return obj

def render_wireframe(obj):
    # 获取相机模型的材质
    material = obj.data.materials[0]
    material.use_nodes = True
    nodes = material.node_tree.nodes
    
    # 创建 Wireframe 节点
    wireframe_node = nodes.new(type='ShaderNodeWireframe')
    wireframe_node.use_pixel_size = True  # 设置 wireframe render as pixel size
    wireframe_node.inputs[0].default_value = 8  # 设置 wireframe 宽度为 pixel 8
    
    # 创建 Mix Shader 节点
    mix_shader_node = nodes.new(type='ShaderNodeMixShader')
    mix_shader_node.inputs['Fac'].default_value = 0.5  # 设置 Mix Shader 的 Fac 为 1.0
    
    # 创建 RGB 节点
    rgb_node = nodes.new(type='ShaderNodeRGB')
    rgb_node.outputs[0].default_value = (0.1, 0.1, 0.1, 1)  # 设置颜色为 gray
    
    # 创建 Principled BSDF 节点
    principled_bsdf = nodes.get("Principled BSDF")
    
    # 将 Wireframe 的 Fac 输入链接到 Mix Shader 的 Fac
    material.node_tree.links.new(wireframe_node.outputs['Fac'], mix_shader_node.inputs['Fac'])
    
    # 将 Principled BSDF 和 Wireframe 分别链接到 Mix Shader 的两个 Shader 输入
    material.node_tree.links.new(principled_bsdf.outputs['BSDF'], mix_shader_node.inputs[1])
    material.node_tree.links.new(rgb_node.outputs['Color'], mix_shader_node.inputs[2])
    
    # 设置 Mix Shader 作为材质的输出节点
    material.node_tree.links.new(mix_shader_node.outputs['Shader'], material.node_tree.nodes['Material Output'].inputs['Surface'])

def transform_obj(obj, rot, trans):

    rot = rot.tolist()
    trans = trans.tolist()
    
    rot = mathutils.Matrix(rot)
    trans = mathutils.Vector(trans)
    
    transform_matrix = mathutils.Matrix.Translation(trans).to_4x4() @ rot.to_4x4()
    
    # 应用变换矩阵到相机对象
    obj.matrix_world = transform_matrix
    
def extrinsic_to_blender_camera(extrinsic, camera_name="Camera", fov=75, collection_name="Cameras"):
    # 将FOV转换为焦距
    def fov_to_focal_length(fov, sensor_width):
        focal_length = (sensor_width / 2) / np.tan(np.radians(fov) / 2)
        return focal_length

    # 输入的相机内参
    sensor_width = 36  # 35mm 传感器宽度

    # 将外参转换为相机位姿矩阵
    camera_matrix = extrinsic_to_cam(extrinsic)

    # 检查指定的 Collection 是否已存在
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[collection_name]


    if camera_name in bpy.context.scene.objects:
        camera_object = bpy.data.objects[camera_name] 
    else:
        if camera_name in bpy.data.objects:
            obj = bpy.data.objects[camera_name] 
            # 删除物体数据
            bpy.data.objects.remove(obj, do_unlink=True)

        # 创建相机对象
        camera_data = bpy.data.cameras.new(camera_name)
        camera_object = bpy.data.objects.new(camera_name, camera_data)   

        # 设置相机的位置和旋转
        camera_object.matrix_world = mathutils.Matrix(camera_matrix)    

        # 设置相机的焦距和传感器宽度
        camera_data.lens = fov_to_focal_length(fov, sensor_width)
        camera_data.sensor_width = sensor_width

        # 将相机添加到指定的 Collection 中  
        new_collection.objects.link(camera_object)

    # 返回相机对象
    return camera_object

def extrinsic_to_cam(extrinsic):
    cam = np.eye(4)
    cam[:3, :3] = extrinsic[:3, :3].T @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    cam[:3, 3] = -(extrinsic[:3, :3].T @ extrinsic[:3, 3])
    return cam

def register():
    bpy.types.Scene.sloper4d_file_path = bpy.props.StringProperty(
        subtype="FILE_PATH"
        )
    bpy.types.Scene.sloper4d_frame_number = bpy.props.IntProperty()
    bpy.types.Scene.point_cube_size = bpy.props.FloatProperty(
        description="The cube's length (cm) of point clouds",
        default = 2.0,
        # unit='cm'
        )
    bpy.types.Scene.execute_function = bpy.props.BoolProperty(
        default=True
        )
    
    bpy.types.Scene.my_start_frame = bpy.props.IntProperty(name="Start Frame")
    bpy.types.Scene.my_end_frame = bpy.props.IntProperty(name="End Frame")
    bpy.types.Scene.my_output_folder = bpy.props.StringProperty(
        name="Output Folder",
        default="/tmp/",
        subtype="DIR_PATH")

    bpy.utils.register_class(Sloper4DViewerPanel)
    bpy.utils.register_class(Sloper4DViewerLoadDataOperator)
    bpy.utils.register_class(Sloper4DViewerSetDataOperator)
    bpy.utils.register_class(Sloper4DViewerEraseDataOperator)
    bpy.utils.register_class(SetOptVisibleSMPLOperator)
    bpy.utils.register_class(SetIMUVisibleSMPLOperator)
    bpy.utils.register_class(SetBaseline2VisibleSMPLOperator)
    bpy.utils.register_class(MyRenderOperator)
    bpy.app.handlers.frame_change_post.append(execute_on_frame_change)
    
def unregister():
    del bpy.types.Scene.sloper4d_file_path
    del bpy.types.Scene.sloper4d_frame_number
    del bpy.types.Scene.point_cube_size
    del bpy.types.Scene.execute_function
    del bpy.types.Scene.my_start_frame
    del bpy.types.Scene.my_end_frame
    del bpy.types.Scene.my_output_folder

    bpy.utils.unregister_class(Sloper4DViewerPanel)
    bpy.utils.unregister_class(Sloper4DViewerLoadDataOperator)
    bpy.utils.unregister_class(Sloper4DViewerSetDataOperator)
    bpy.utils.unregister_class(Sloper4DViewerEraseDataOperator)
    bpy.utils.unregister_class(SetOptVisibleSMPLOperator)
    bpy.utils.unregister_class(SetIMUVisibleSMPLOperator)
    bpy.utils.unregister_class(SetBaseline2VisibleSMPLOperator)
    bpy.utils.unregister_class(MyRenderOperator)

    bpy.app.handlers.frame_change_post.remove(execute_on_frame_change)

if __name__ == "__main__":
    register()
