# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import cv2
from typing import List
import pycocotools.mask as mask_util

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    # _create_text_labels,
)
from detectron2.utils.colormap import random_color, random_colors


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


class HumanVisualizer:
    def __init__(self, cfg_file, instance_mode=ColorMode.IMAGE):
        """ mininum visualizer for COCO17
        """
        cfgs = get_cfg() # default configuration
        cfgs.merge_from_file(cfg_file)
        self.metadata = MetadataCatalog.get(
            cfgs.DATASETS.TEST[0] if len(cfgs.DATASETS.TEST) else "__unused"
        )
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            # ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode
        self._max_num_instances = self.metadata.get("max_num_instances", 74)
        self._assigned_colors = {}
        self._color_pool = random_colors(self._max_num_instances, rgb=True, maximum=1)
        self._color_idx_set = set(range(len(self._color_pool)))

    def draw_predicted_humans(self, frame, ids=None, bboxs=None, kps=None):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(ids)
        if num_instances == 0:
            return frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_visualizer = Visualizer(frame, self.metadata)

        if ids is not None:
            colors = self._assign_colors_by_id(ids)
        else:
            # TODO: clean old assign color method and use a default tracker to assign id
            # FIXME: human_cls_id = 0
            detected = [
                _DetectedInstance(0, bboxs[i], mask_rle=None, color=None, ttl=8)
                for i in range(num_instances)
            ]
            colors = self._assign_colors(detected)
        alpha = 0.5
        # labels = _create_text_labels([self.cfg.human_cls_id for _ in range(num_instances)], None, self.metadata.get("thing_classes", None))
        labels = ["ID_%06d"%(ids[i]) for i in range(num_instances)]
        frame_visualizer.overlay_instances(
            boxes=bboxs,
            masks=None,
            keypoints=kps,
            labels=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        im = cv2.cvtColor(frame_visualizer.output.get_image(), cv2.COLOR_RGB2BGR)
        return im

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        if instances[0].bbox is None:
            assert instances[0].mask_rle is not None
            # use mask iou only when box iou is None
            # because box seems good enough
            rles_old = [x.mask_rle for x in self._old_instances]
            rles_new = [x.mask_rle for x in instances]
            ious = mask_util.iou(rles_old, rles_new, is_crowd)
            threshold = 0.5
        else:
            boxes_old = [x.bbox for x in self._old_instances]
            boxes_new = [x.bbox for x in instances]
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
            threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]

    def _assign_colors_by_id(self, idx) -> List:
        colors = []
        untracked_ids = set(self._assigned_colors.keys())
        for id in idx:
            if id in self._assigned_colors:
                colors.append(self._color_pool[self._assigned_colors[id]])
                untracked_ids.remove(id)
            else:
                assert (
                    len(self._color_idx_set) >= 1
                ), f"Number of id exceeded maximum, \
                    max = {self._max_num_instances}"
                idx = self._color_idx_set.pop()
                color = self._color_pool[idx]
                self._assigned_colors[id] = idx
                colors.append(color)
        for id in untracked_ids:
            self._color_idx_set.add(self._assigned_colors[id])
            del self._assigned_colors[id]
        return colors
