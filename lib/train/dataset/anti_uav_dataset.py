import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from lib.train.data import jpeg4py_loader
from lib.train.dataset.base_video_dataset import BaseVideoDataset


class AntiUAVDataset(BaseVideoDataset):
    def __init__(self, root, split="train", image_loader=jpeg4py_loader):
        assert split in ["train", "validation"]
        root = Path(root) / split
        super(AntiUAVDataset, self).__init__("Anti-UAV", root, image_loader)
        self.sequence_list = [s.name for s in root.iterdir()]
        self.seq_per_class = {"drone": self.sequence_list}
        self.class_list = ["drone"]

    def get_name(self):
        return "anti-uav"

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def get_sequence_info(self, seq_id):
        bboxes, valid, visible = self._load_ann(seq_id)
        return {"bbox": bboxes, "valid": valid, "visible": visible}

    def _load_ann(self, seq_id):
        sequence_name = self.sequence_list[seq_id]
        ann_path = self.root / sequence_name / "IR_label.json"
        with open(str(ann_path), "r") as f:
            anns = json.load(f)

        visible = torch.tensor(anns["exist"])
        valid = visible.clone()

        bboxes = anns["gt_rect"]
        bboxes_true = torch.empty((0, 4), dtype=torch.float32)
        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4 or sum(bbox) == 0 or bbox[2] == 0 or bbox[3] == 0:
                valid[i] = 0
                bbox_true = torch.zeros((1, 4))
            else:
                bbox_true = torch.tensor(bbox).reshape(1, 4)
            bboxes_true = torch.cat([bboxes_true, bbox_true])
        return bboxes_true, valid, visible

    def get_class_name(self, seq_id):
        return "drone"

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_class = self.get_class_name(seq_id)

        object_meta = OrderedDict(
            {
                "object_class_name": obj_class,
                "motion_class": None,
                "major_class": None,
                "root_class": None,
                "motion_adverb": None,
            }
        )
        return frame_list, anno_frames, object_meta

    def _get_frame(self, seq_id, frame_id):
        sequence_name = self.sequence_list[seq_id]
        image_path = self.root / sequence_name / (str(frame_id + 1).zfill(6) + ".jpg")
        image = self.image_loader(str(image_path))
        return image
