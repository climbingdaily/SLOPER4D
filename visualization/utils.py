import os
import math
import sys
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
sys.path.append('..')
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
    if filterx: pc = pc[pc[:, 0] > 0]
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
    rgb = np.array([0,0,1.])
    for i, (x, y) in enumerate(sence_pc):
        x = int(math.floor(x))
        y = int(math.floor(y))
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        cv2.circle(image, (x, y), 2, color=rgb*255, thickness=-1)
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
