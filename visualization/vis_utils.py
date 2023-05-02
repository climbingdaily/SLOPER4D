import os
import math
import sys
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pycocotools import mask as mask_utils

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

sys.path.append(root_folder)
from utils import pypcd

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]
BONES = [    
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [7, 9], [6, 8],
    [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

COCO_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170)
]

JOINTS = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 
5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 
9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 
13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def plot_coco_annotation(img: np.ndarray,
                         keypoints: Optional[np.ndarray] = None,
                         bboxes: Optional[Tuple[int, int, int, int]] = None,
                         mask: Optional[List[dict]] = None,
                         keypoint_radius: int = 3,
                         line_width: int = 2,
                         alpha: float = 0.7,
                         text: str='',
                         _KEYPOINT_THRESHOLD: Optional[float] = 0.05,
                         save_path: Optional[str] = None) -> np.ndarray:
    overlay = np.copy(img)

    if mask is not None and len(mask) > 0:
        coordinate = np.array(mask)
        masks = get_bool_array_from_coordinates(coordinate)[None, :, :]
        mask_image = load_mask(masks, False)
        img = cv2.add(img, mask_image)
        
        # for m in mask:
        #     poly = mask_utils.decode(m['segmentation'])
        #     color = np.random.rand(3) * 255
        #     cv2.fillPoly(img, [poly], color=color.astype(int))
    
    if bboxes is not None and len(bboxes) > 0:
        for bbox in bboxes:
            if len(bbox) > 0:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            color=(220, 173, 69), thickness=3)
            
    if keypoints is not None and len(keypoints) > 0:
        for per_kpt in keypoints:
            if len(per_kpt) == 0:
                continue
            per_kpt    = per_kpt.reshape(-1, 3)
            points     = per_kpt[:, :2].astype(int)
            visibility = per_kpt[:, 2]
                    
            for i, conn in enumerate(BONES):
                if visibility[conn[0]] > _KEYPOINT_THRESHOLD[i] and visibility[conn[1]] > _KEYPOINT_THRESHOLD[i]:
                    cv2.line(img, tuple(points[conn[0]]), tuple(points[conn[1]]), 
                             color=(np.array(COCO_COLORS[conn[0]]) + np.array(COCO_COLORS[conn[1]]))/2, 
                             thickness=line_width)
                else:
                    cv2.line(img, tuple(points[conn[0]]), tuple(points[conn[1]]), 
                             color=(100, 100, 100), 
                             thickness=line_width-1)
                    
            cv2.addWeighted(overlay, 1-alpha, img, alpha, 0, img)

            for i, p in enumerate(points):
                if visibility[i] > _KEYPOINT_THRESHOLD[i]:
                    cv2.circle(img, (p[0], p[1]), 
                               radius=keypoint_radius, 
                               color=COCO_COLORS[i], 
                               thickness=-1)
                else:
                    cv2.circle(img, (p[0], p[1]), 
                               radius=keypoint_radius-1, 
                               color=(100,100,100), 
                               thickness=-1)

    if text is not None and len(text) > 0:
        cv2.putText(img, os.path.basename(text), (30, 60), DEFAULT_FONT, 1, BLACK, 2)

    if save_path is not None:
        cv2.imwrite(save_path, img)
        
    return img


def draw_bbox(img, box, cls_name, identity=None, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img


def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = COLORS_10[id%len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def boxes_xyxy2xcycwh(xyxy):
    xcycwh = np.array([
        (xyxy[:,0]+xyxy[:,2]) / 2,
        (xyxy[:,1]+xyxy[:,3]) / 2,
        xyxy[:,2] - xyxy[:,0],
        xyxy[:,3] - xyxy[:,1]
    ]).T
    return xcycwh


def load_mask(mask, random_color):

    if random_color:
        color = np.random.rand(3) * 255
    else:
        color = np.array([0, 100, 100])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    return mask_image

def load_box(box, image):

    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    return image

def get_bool_array_from_coordinates(coordinates, shape=(1080, 1920)):
    bool_arr = np.zeros(shape, dtype=bool)
    if len(coordinates) > 0:
        bool_arr[coordinates[:, 0], coordinates[:, 1]] = True

    return bool_arr

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def extrinsic_to_cam(extrinsic):
    cam = np.eye(4)
    cam[:3, :3] = extrinsic[:3, :3].T @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    cam[:3, 3] = -(extrinsic[:3, :3].T @ extrinsic[:3, 3])
    return cam

def cam_to_extrinsic(cam):
    """
    It takes a camera matrix and returns the extrinsic matrix
    
    Args:
      cam: the camera matrix
    
    Returns:
      The extrinsic matrix
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ cam[:3, :3].T
    extrinsic[:3, 3] = -(extrinsic[:3, :3] @ cam[:3, 3])
    return extrinsic

### point cloud ###

def read_pcd_with_color(pcd_file, color_fmt="depth", filterx=True):
    """
    Read pcd file
        return columns: x, y, z, rgb
    Refer to:
        https://github.com/climbingdaily/LidarHumanScene/blob/master/tools/tool_func.py
    """
    pc_pcd = pypcd.PointCloud.from_path(pcd_file)
    pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
    pc[:, 0] = pc_pcd.pc_data['x']
    pc[:, 1] = pc_pcd.pc_data['y']
    pc[:, 2] = pc_pcd.pc_data['z']
    if color_fmt == "intensity":
        if 'intensity' not in pc_pcd.fields: assert(0)
        intensity = pc_pcd.pc_data['intensity'].reshape(-1)
        if np.max(pc_pcd.pc_data['intensity']) > 255:
            intensity = np.array([
                155 * np.log2(i/100) / np.log2(864) + 100 if i > 100 else i \
                for i in intensity
            ])
        rgb = plt.get_cmap('hsv')(intensity/255)[:, :3]
    elif color_fmt == "depth":
        depth = np.sqrt(np.sum(pc*pc, axis=1))
        depth = depth / (np.sum(depth)/depth.shape[0]) * 255
        depth = np.array([
                155 * np.log2(i/100) / np.log2(864) + 200 if i > 200 else i \
                for i in depth
            ])
        rgb = plt.get_cmap('hsv')(depth/255)[:, :3]
    elif color_fmt == "normal":
        # TODO
        pass
    elif 'rgb' in pc_pcd.fields:
        rgb = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb'])/255
    assert(rgb is not None), "RGB is None!!"
    pc = np.concatenate((pc, rgb), axis=1)
    # if filterx: pc = pc[pc[:, 1] > 0]
    return pc

def read_pcd(pcd_file, color_fmt="depth"):
    """
    Read pcd file
        return columns: x, y, z, rgb
    Refer to:
        https://github.com/climbingdaily/LidarHumanScene/blob/master/tools/tool_func.py
    """
    pc_pcd = pypcd.PointCloud.from_path(pcd_file)
    pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
    pc[:, 0] = pc_pcd.pc_data['x']
    pc[:, 1] = pc_pcd.pc_data['y']
    pc[:, 2] = pc_pcd.pc_data['z']
    return pc

def camera_to_pixel(X, intrinsics, distortion_coefficients):
    # focal length
    f = intrinsics[:2]
    # center principal point
    c = intrinsics[2:]
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / (X[..., 2:])
    # XX = pd.to_numeric(XX, errors='coere')
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c

def draw_pc2image(image, sence_pc, colors):
    for i, (x, y) in enumerate(sence_pc):
        rgb = colors[i]
        x = int(math.floor(x))
        y = int(math.floor(y))
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        cv2.circle(image, (x, y), 1, color=rgb*255, thickness=-1)
    return image

def draw_skel2image(image, sence_pc):
    for i, (x, y) in enumerate(sence_pc):
        x = int(math.floor(x))
        y = int(math.floor(y))
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        cv2.circle(image, (x, y), 3, color=COCO_COLORS[i], thickness=-1)
    return image

def draw_pc2mask(mask, pc, size=1):
    def _valid(y, x):
        return not (x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0])
    def _draw_chunk(x, y):
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                if _valid(x+dx, y+dy):
                    mask[y+dx][x+dy] = True
    for x, y in pc:
        x = np.floor(x).astype(int)
        y = np.floor(y).astype(int)
        if _valid(x, y): _draw_chunk(x, y)
    return mask


def world_to_camera(X, extrinsic_matrix):
    n = X.shape[0]
    X = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    X = np.dot(extrinsic_matrix, X).T
    return X[..., :3]

def plot_points_on_img(img, 
                       points3d, 
                       extrinsic, 
                       intrinsic,
                       img_path='',
                       dist=np.zeros(5), 
                       colors=None, 
                       max_depth=15, 
                       save_img=False):
    """
    This function takes in an image, 3D points, camera extrinsic and intrinsic parameters, and projects
    the 3D points onto the image and saves the resulting image.
    
    Args:
      img_path: The file path of the image on which the points will be plotted.
      points3d: A numpy array of shape (N, 3) representing N 3D points in world coordinates.
      extrinsic: The extrinsic matrix represents the position and orientation of the camera in the world
    coordinate system. It is a 3x4 matrix that combines the rotation and translation of the camera.
      intrinsic: The intrinsic matrix of the camera, which contains information about the focal length,
    principal point, and skew. It is used to convert 3D points in camera coordinates to 2D pixel
    coordinates.
      dist: The distortion coefficients of the camera used to capture the image.
      colors: An optional array of colors for each point in the 3D space. If provided, the colors will
    be used to color the points in the image. If not provided, a default color map will be used based on
    the depth of each point.
      max_depth: The maximum depth value for the points to be plotted. Points with depth values greater
    than this will not be plotted. Defaults to 15
    """

    camera_points = world_to_camera(points3d, extrinsic)
    if colors is not None:
        colors = colors[camera_points[:, 2] > 0]
    camera_points = camera_points[camera_points[:, 2] > 0]
    pixel_points  = camera_to_pixel(camera_points, intrinsic, dist)
    pixel_points  = np.round(pixel_points).astype(np.int32)

    rule1 = pixel_points[:, 0] >= 0
    rule2 = pixel_points[:, 0] < img.shape[1]
    rule3 = pixel_points[:, 1] >= 0
    rule4 = pixel_points[:, 1] < img.shape[0]
    rule  = [a and b and c and d for a, b, c, d in zip(rule1, rule2, rule3, rule4)]

    camera_points = camera_points[rule]
    pixel_points  = pixel_points[rule]
    depth         = np.linalg.norm(camera_points, axis=1)
    
    if colors is not None:
        colors = colors[rule]
    else:
        colors = plt.get_cmap('hsv')(depth / max_depth)[:, :3] * 255

    for d, color, (x, y) in zip(depth, colors, pixel_points):
        if d > 0.5:
            cv2.circle(img, (x, y), 1, color=color, thickness=-1)
            
    if save_img:
        save_img_path = f"{os.path.splitext(img_path)[0]}_proj.jpg"
        cv2.imwrite(save_img_path, img)
        print(f"Image saved to {save_img_path}")

        img_title = 'Points overlay'

        cv2.imshow(img_title, img)
        while cv2.getWindowProperty(img_title, cv2.WND_PROP_VISIBLE) > 0:
            cv2.imshow(img_title, img)
            key = cv2.waitKey(100)
            if key == 27:  # Press 'Esc'
                break

    return img