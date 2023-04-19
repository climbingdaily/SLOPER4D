# -*- coding: utf-8 -*-
# @Author  : Zhang.Jingyi

"""
This file contains definitions of useful data structures and the paths 
for the datasets and data files necessary to run the code.
"""

import os
import sys
from os.path import join

SMPL_DIR = os.path.split(os.path.abspath( __file__))[0]

SMPL_NEUTRAL = os.path.join(SMPL_DIR, 'SMPL_NEUTRA.pkl')
SMPL_FEMALE = os.path.join(SMPL_DIR, 'SMPL_FEMALE.pkl')
SMPL_MALE = os.path.join(SMPL_DIR, 'SMPL_MALE.pkl')

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DIR, 'J_regressor_extra.npy')

SMPL_SAMPLE_PLY = os.path.join(SMPL_DIR, 'smpl_sample.ply')

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""

JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18,
              20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

COL_NAME = [
    "time",
    "Hips.y",
    "Hips.x",
    "Hips.z",
    "RightUpLeg.y",
    "RightUpLeg.x",
    "RightUpLeg.z",
    "RightLeg.y",
    "RightLeg.x",
    "RightLeg.z",
    "RightFoot.y",
    "RightFoot.x",
    "RightFoot.z",
    "LeftUpLeg.y",
    "LeftUpLeg.x",
    "LeftUpLeg.z",
    "LeftLeg.y",
    "LeftLeg.x",
    "LeftLeg.z",
    "LeftFoot.y",
    "LeftFoot.x",
    "LeftFoot.z",
    "Spine.y",
    "Spine.x",
    "Spine.z",
    "Spine1.y",
    "Spine1.x",
    "Spine1.z",
    "Spine2.y",
    "Spine2.x",
    "Spine2.z",
    "Neck.y",
    "Neck.x",
    "Neck.z",
    "Neck1.y",
    "Neck1.x",
    "Neck1.z",
    "Head.y",
    "Head.x",
    "Head.z",
    "RightShoulder.y",
    "RightShoulder.x",
    "RightShoulder.z",
    "RightArm.y",
    "RightArm.x",
    "RightArm.z",
    "RightForeArm.y",
    "RightForeArm.x",
    "RightForeArm.z",
    "RightHand.y",
    "RightHand.x",
    "RightHand.z",
    "RightHandThumb1.y",
    "RightHandThumb1.x",
    "RightHandThumb1.z",
    "RightHandThumb2.y",
    "RightHandThumb2.x",
    "RightHandThumb2.z",
    "RightHandThumb3.y",
    "RightHandThumb3.x",
    "RightHandThumb3.z",
    "RightInHandIndex.y",
    "RightInHandIndex.x",
    "RightInHandIndex.z",
    "RightHandIndex1.y",
    "RightHandIndex1.x",
    "RightHandIndex1.z",
    "RightHandIndex2.y",
    "RightHandIndex2.x",
    "RightHandIndex2.z",
    "RightHandIndex3.y",
    "RightHandIndex3.x",
    "RightHandIndex3.z",
    "RightInHandMiddle.y",
    "RightInHandMiddle.x",
    "RightInHandMiddle.z",
    "RightHandMiddle1.y",
    "RightHandMiddle1.x",
    "RightHandMiddle1.z",
    "RightHandMiddle2.y",
    "RightHandMiddle2.x",
    "RightHandMiddle2.z",
    "RightHandMiddle3.y",
    "RightHandMiddle3.x",
    "RightHandMiddle3.z",
    "RightInHandRing.y",
    "RightInHandRing.x",
    "RightInHandRing.z",
    "RightHandRing1.y",
    "RightHandRing1.x",
    "RightHandRing1.z",
    "RightHandRing2.y",
    "RightHandRing2.x",
    "RightHandRing2.z",
    "RightHandRing3.y",
    "RightHandRing3.x",
    "RightHandRing3.z",
    "RightInHandPinky.y",
    "RightInHandPinky.x",
    "RightInHandPinky.z",
    "RightHandPinky1.y",
    "RightHandPinky1.x",
    "RightHandPinky1.z",
    "RightHandPinky2.y",
    "RightHandPinky2.x",
    "RightHandPinky2.z",
    "RightHandPinky3.y",
    "RightHandPinky3.x",
    "RightHandPinky3.z",
    "LeftShoulder.y",
    "LeftShoulder.x",
    "LeftShoulder.z",
    "LeftArm.y",
    "LeftArm.x",
    "LeftArm.z",
    "LeftForeArm.y",
    "LeftForeArm.x",
    "LeftForeArm.z",
    "LeftHand.y",
    "LeftHand.x",
    "LeftHand.z",
    "LeftHandThumb1.y",
    "LeftHandThumb1.x",
    "LeftHandThumb1.z",
    "LeftHandThumb2.y",
    "LeftHandThumb2.x",
    "LeftHandThumb2.z",
    "LeftHandThumb3.y",
    "LeftHandThumb3.x",
    "LeftHandThumb3.z",
    "LeftInHandIndex.y",
    "LeftInHandIndex.x",
    "LeftInHandIndex.z",
    "LeftHandIndex1.y",
    "LeftHandIndex1.x",
    "LeftHandIndex1.z",
    "LeftHandIndex2.y",
    "LeftHandIndex2.x",
    "LeftHandIndex2.z",
    "LeftHandIndex3.y",
    "LeftHandIndex3.x",
    "LeftHandIndex3.z",
    "LeftInHandMiddle.y",
    "LeftInHandMiddle.x",
    "LeftInHandMiddle.z",
    "LeftHandMiddle1.y",
    "LeftHandMiddle1.x",
    "LeftHandMiddle1.z",
    "LeftHandMiddle2.y",
    "LeftHandMiddle2.x",
    "LeftHandMiddle2.z",
    "LeftHandMiddle3.y",
    "LeftHandMiddle3.x",
    "LeftHandMiddle3.z",
    "LeftInHandRing.y",
    "LeftInHandRing.x",
    "LeftInHandRing.z",
    "LeftHandRing1.y",
    "LeftHandRing1.x",
    "LeftHandRing1.z",
    "LeftHandRing2.y",
    "LeftHandRing2.x",
    "LeftHandRing2.z",
    "LeftHandRing3.y",
    "LeftHandRing3.x",
    "LeftHandRing3.z",
    "LeftInHandPinky.y",
    "LeftInHandPinky.x",
    "LeftInHandPinky.z",
    "LeftHandPinky1.y",
    "LeftHandPinky1.x",
    "LeftHandPinky1.z",
    "LeftHandPinky2.y",
    "LeftHandPinky2.x",
    "LeftHandPinky2.z",
    "LeftHandPinky3.y",
    "LeftHandPinky3.x",
    "LeftHandPinky3.z"
]

bones=[(0,1), (1,4), (4,7), (7,10), # R leg
       (0,2), (2,5), (5,8), (8,11), # L leg
       (0,3), (3,6), (6,9), # Spine
       (9,12), (12,15), # Head
       (9,13), (13,16), (16,18), (18,20), (20,22), # R arm
       (9,14), (14,17), (17,19), (19,21), (21,23)] # L arm

body_parts = {
    'heads': [12, 15],
    'torso': [3, 6, 9, 13, 14],
    'legs' : [1, 2, 4, 5, 7, 8],
    'arms' : [16, 17, 18, 19, 20, 21],
    'ends' : [10, 11, 22, 23]
}