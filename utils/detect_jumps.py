################################################################################
# File: /get_mocap_trans_from_bvh.py                                           #
# Created Date: Monday June 27th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

from cProfile import label
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from pandas import describe_option
from scipy.spatial.transform import Rotation as R

from bvhtoolbox import Bvh  # pip install bvhtoolbox

sys.path.append(os.path.dirname(os.path.split(os.path.abspath(__file__))[0]))
from utils import save_json_file, read_json_file, MOCAP_INIT, load_csv_data


def detect_jump(left_foot, right_foot, prominences = 0.2, width = 100):
    """
    > It finds the peaks in the left and right foot data, and then finds the peaks that are close to
    each other in time, and have a large enough prominence
    
    Args:
      left_foot: the left foot's height
      right_foot: the right foot data
      prominences: the minimum height of the peak, relative to the surrounding troughs.
      width: the width of the window to look for the minimum value in. Defaults to 100
    """
    from scipy.signal import find_peaks

    lf_height = np.asarray(left_foot[:50]).mean()
    rf_height = np.asarray(right_foot[:50]).mean()
    left_foot = np.asarray(left_foot- lf_height)
    right_foot = np.asarray(right_foot- rf_height)

    l_peaks, lprop = find_peaks(left_foot, distance=80, height=0.05, prominence=0.05)
    r_peaks, rprop = find_peaks(right_foot, distance=80, height=0.05, prominence=0.05)
    jumps = []
    j = 0
    for i, lp in enumerate(l_peaks):
        if j >= len(r_peaks):
            break

        while r_peaks[j] < lp and abs(r_peaks[j] - lp) >= 5:
            if j+1 < len(r_peaks):
                j += 1
            else:
                break

        # two peaks at the same time(< 0.05s), distance < 0.05m, prominences > 0.2m
        if abs(r_peaks[j] - lp) < 5 :  
            peaks_dist = abs(lprop['peak_heights'][i] - rprop['peak_heights'][j])
            # peak_prominences =  (lprop['prominences'][i] + rprop['prominences'][j])/2
            wl = max(0, lp - width)
            wr = min(lp+width, left_foot.shape[0])
            peak_prominences = (lprop['peak_heights'][i] - left_foot[wl:wr].min() + rprop['peak_heights'][j] - right_foot[wl:wr].min())/2

            if peaks_dist < 0.05 and peak_prominences > prominences:
                jumps.append(lp)
    return l_peaks, r_peaks, jumps


def loadjoint(frames, joint_index, frame_time=0.01):
    """
    It takes in the joint data, and returns the joint data in the format that we want
    
    Args:
      frames: the data from the csv file
      joint_index: the index of the joint you want to extract.
      frame_time: the time between each frame, in seconds.
    
    Returns:
      the joint data in the form of a numpy array.
    """
    joint = frames[:, joint_index * 6:joint_index*6 + 3] * frame_time
    joint_rot = frames[:, joint_index * 6 + 3:joint_index*6 + 6]
    rot = R.from_euler('yxz', joint_rot, degrees=True)
    frame_number = np.arange(frames.shape[0])
    frame_number = frame_number.reshape((-1, 1))
    frame_time = frame_number * frame_time

    # from mocap coordinate to Velodyne coordinate, make the Z-axis point up
    joint = joint @ MOCAP_INIT.T
    rot = (R.from_matrix(MOCAP_INIT) * rot).as_quat()

    save_joint = np.concatenate(
        (frame_number, joint, rot, frame_time), axis=-1)
    return save_joint

def get_mocap_root(mpcap_file, save_dir = None, joint_index=0):
    """
    > The function takes in a bvh file and returns the root joint's translation in each frame
    
    Args:
      mpcap_file: the path to the bvh file
      save_dir: the directory where you want to save the mocap translation file.
      joint_index: the index of the joint you want to extract. Defaults to 0
    
    Returns:
      The root joint of the mocap data, and the frame rate of the mocap data.
    """
    with open(mpcap_file) as f:
        mocap = Bvh(f.read())

    frame_time = mocap.frame_time
    frames = mocap.frames
    frames = np.asarray(frames, dtype='float32')

    root = loadjoint(frames, joint_index, frame_time)  # unit：meter

    if save_dir:
        save_file = os.path.join(save_dir, 'mocap_trans.txt')
        field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
        np.savetxt(save_file, root, fmt=field_fmts)
        print('Save mocap translation in: ', save_file)
    return root, round(1/frame_time)

def fit_quadratic(x, y, a=-4.916):
    """
    拟合一元二次方程 y = a * (x - t)**2 + h，其中 a 默认为 -1/2g, g=9.88。

    参数：
    x: 一个一维的numpy数组，表示自变量的取值。
    y: 一个一维的numpy数组，表示因变量的取值。
    a: 二次项系数，默认为 -1/2。

    返回值：
    一个元组，包含两个元素：(t, h)。其中 t 表示顶点的横坐标，h 表示顶点的纵坐标。
    """
    from scipy.optimize import curve_fit
    def quadratic(x, t, h):
        return a * (x - t)**2 + h
    popt, popv = curve_fit(quadratic, x, y)
    errors = np.sqrt(np.diag(popv))
    t, h = popt[0], popt[1]
    return t, h, errors

# 画出散点图、拟合曲线和对称轴
def add_plot_fitting(ax, x, y, errors, t, h, a=-4.916):
    # 生成一些随机数据
    x_fit = np.linspace(x.min()-0.05, x.max()+0.05, 100)

    # 拟合一元二次方程
    y_fit = a * (x_fit - t)**2 + h

    # 计算交叉点的坐标
    x_intersect = t
    y_intersect = h

    ax.scatter(x, y, label='data')
    ax.plot(x_fit, y_fit, color='r', label='parabolic path')
    ax.axvline(x=t, color='gray', linestyle='--')
    ax.axhline(y=h, color='gray', linestyle='--')
    ax.annotate(f'({x_intersect:.3f}, {y_intersect:.3f})', (x_intersect, y_intersect),
                xytext=(x_intersect-0.01, y_intersect-0.04), fontsize=10,
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    ax.legend()

def plot_all_peaks(times, jumps, heights, frame_rate, fit_window):
    n_subplots = len(jumps)

    # 计算最接近平方数的整数作为行数和列数
    if n_subplots < 1:
        num_rows = 1
        num_cols = 1
    else:
        num_rows   = int(np.sqrt(n_subplots - 1)) + 1
        num_cols   = int(np.ceil(n_subplots / num_rows))

    # 动态设置图像大小，长宽尽可能小
    fig_width  = 4 * num_cols
    fig_height = 3 * num_rows
    fig, axs   = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    peaks      = []
    new_jumps  = []
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    for i, ax in enumerate(axs.flat):
        if i < n_subplots:
            lj = jumps[i]

            left  = max(0, lj - round(frame_rate*fit_window/2))
            right = min(len(times), lj + round(frame_rate*fit_window/2))
            hh    = heights[left:right+1]
            tt    =   times[left:right+1]
            peak_t, peak_h, errors = fit_quadratic(tt - int(tt[0]), hh)
            if errors.mean() < 0.0015:
                peaks.append(round(peak_t, 3) + int(tt[0]))
                if abs(peaks[-1] - times[lj]) > 0.5/frame_rate:
                    peaks[-1] = times[lj]
                    print(f"{peaks[-1]:3f} is not a good peak, please carefully check it!!!")
                ax.set_title(f"({i+1}) {peaks[-1]:.3f}")
                add_plot_fitting(ax, tt, hh, errors, peaks[-1], peak_h)
                new_jumps.append(lj)
            else:
                print(f'Fitting error: {errors.mean():.5f}')
        else:
            ax.axis("off")
    if len(new_jumps) <= 0:
        new_jumps = jumps
    # fig.legend(["data", "Parabolic path"])

    fig.text(0.5, 0.04, 'Time(s)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Height(cm)', ha='center', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    return peaks, new_jumps, fig

def load_time(folder, index=None):
    file_list = os.listdir(folder)
    time = [float(f.split('.')[0].replace('_', '.')) for f in file_list]
    time = np.array(sorted(time))
    return time if index is None else time[index]

def main(mpcap_file, json_file=None, frame_rate = 20, fit_window=1/6):
    save_dir=os.path.dirname(mpcap_file)
    
    if json_file is None:
        json_file = os.path.join(os.path.dirname(mpcap_file), 'dataset_params.json')
        
    if not os.path.exists(json_file):
        save_json_file(json_file, {})
        
    init_params = read_json_file(json_file)

    os.makedirs(save_dir, exist_ok=True)

    update_property = []

    if mpcap_file.endswith('.bvh'):
        print('Bvh file: ', mpcap_file)
        root, frame_rate = get_mocap_root(mpcap_file, save_dir=save_dir, joint_index=0)

        _, _, mocap_jumps = detect_jump(root[:, 3], root[:, 3], width=int(frame_rate/2))
        print(mocap_jumps)

        peaks, mocap_jumps, fig = plot_all_peaks(np.arange(len(root))/frame_rate, 
                                    mocap_jumps, 
                                    root[:, 3], 
                                    frame_rate,
                                    fit_window)
        fig.savefig(mpcap_file.replace('.bvh', '_fitting.png'))

        print('Jumps', mocap_jumps, peaks)
        if 'mocap_sync' not in init_params.keys():
            init_params['mocap_sync'] = peaks
            update_property.append('mocap_sync')

        if 'mocap_framerate' not in init_params.keys():
            init_params['mocap_framerate'] = frame_rate
            update_property.append('mocap_framerate')

        if 'first' in mpcap_file and 'first_person' not in init_params.keys():
            update_property.append('first_person:mocap_sync')
            init_params['first_person'] = {'mocap_sync': peaks}

        if 'second' in mpcap_file and 'second_person' not in init_params.keys():
            init_params['second_person'] = {'mocap_sync': peaks}
            update_property.append('second_person:mocap_sync')
    
    elif mpcap_file.endswith('.csv'):
        print('CSV file: ', mpcap_file)
        
        pos_data, pos_data_bak, col_names = load_csv_data(mpcap_file, mpcap_file)

        # col_names = [cc.split('.')[0] for cc in col_names]
        # for i, cc in enumerate(col_names):
        #     if i % 3 == 1:
        #         print((i-1)/3, cc)

        mocap_time = pos_data_bak[:, 0]
        root       = pos_data[:, 0] @ MOCAP_INIT.T
        frame_rate = round(1/(mocap_time[1:] - mocap_time[:-1]).mean())
        save_dir   = os.path.dirname(mpcap_file)

        if save_dir:
            save_file = os.path.join(save_dir, 'mocap_trans.txt')
            np.savetxt(save_file, root, fmt=['%.6f', '%.6f', '%.6f'])
            print('Save mocap translation in: ', save_file)
            
        _, _, mocap_jumps = detect_jump(root[:, 2], root[:, 2], width=int(frame_rate/2))

        print(mocap_jumps)

        peaks, mocap_jumps, fig = plot_all_peaks(np.arange(len(root))/frame_rate, 
                                    mocap_jumps, 
                                    root[:, 2], 
                                    frame_rate,
                                    fit_window)
        fig.savefig(mpcap_file.replace('.csv', '_fitting.png'))

        print('Jumps', mocap_jumps, peaks)
        if 'mocap_sync' not in init_params.keys():
            init_params['mocap_sync'] = peaks
            update_property.append('mocap_sync')

        if 'mocap_framerate' not in init_params.keys():
            init_params['mocap_framerate'] = frame_rate
            update_property.append('mocap_framerate')

        if 'first' in mpcap_file and 'first_person' not in init_params.keys():
            update_property.append('first_person:mocap_sync')
            init_params['first_person'] = {'mocap_sync': peaks}

        if 'second' in mpcap_file and 'second_person' not in init_params.keys():
            init_params['second_person'] = {'mocap_sync': peaks}
            update_property.append('second_person:mocap_sync')

    elif mpcap_file.endswith('.txt'):
        print('LiDAR traj file', mpcap_file)

        lidar_time      = np.loadtxt(mpcap_file)[:, -1]
        root            = np.loadtxt(mpcap_file)[:, 1:4]
        frame_rate      = round(1 / np.diff(lidar_time).mean())

        _, _, lidar_jumps = detect_jump(
            root[:, 2], root[:, 2], width=int(frame_rate/2))

        peaks, lidar_jumps, fig = plot_all_peaks(lidar_time, lidar_jumps, root[:, 2], frame_rate, fit_window)
        fig.savefig(mpcap_file.replace('.txt', '_fitting.png'))

        print('lidar jumps', lidar_jumps, peaks)

        if 'lidar_sync' not in init_params.keys():
            init_params['lidar_sync'] = peaks 
            update_property.append('lidar_sync')

        if 'lidar_framerate' not in init_params.keys():
            update_property.append('lidar_framerate')
            init_params['lidar_framerate'] = frame_rate
    else:
        print("File format error!!!!")
        exit(0)

    if 'lidar_sync' not in init_params.keys():
        init_params['lidar_sync'] = []

    if 'lidar_framerate' not in init_params.keys():
        init_params['lidar_framerate'] = []
    
    save_json_file(json_file, init_params)

    print('==============================')
    if len(update_property) == 0:
        print(f'All properties are already existed.')
        print(f'Please manually update them.')
    else:
        print("These properties are added")
        print(update_property)
    print('==============================')
    
if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    print('GET BVH ROOT POSITIONS......')

    parser.add_argument("-M", "--mpcap_file", type=str, default=None)

    parser.add_argument("-J", "--json_file", type=str, default=None, 
                        help='Where to save the information')
                        
    parser.add_argument("-FR", "--framerate", type=float, default=20, 
                        help='The framerate of the sensor data')

    parser.add_argument("-FW", "--fit_window", type=float,
                        default=0.2, help='The time window to fit the curve.')

    args = parser.parse_args()

    main(args.mpcap_file, args.json_file, args.framerate, args.fit_window)
