import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data
import glob

logger = logging.getLogger(__name__)


class S1PTDataset(PTBaseDataset):

    def __init__(self, ann_file, **kwargs):
        super().__init__()
        self.feat_file, self.attribute_file, self.anno_file = ann_file[:3]

        self.feats = torch.load(self.feat_file)
        self.attributes = json.load(open(self.attribute_file, 'r'))
        self.anno = json.load(open(self.anno_file, 'r'))

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, obj_id, scene_feat, scene_attr = self.get_anno(index)
        target_captions = self.anno[index]["captions"]
        return scene_feat, scene_attr, obj_id, target_captions


def s1_collate_fn(batch):
    scene_feats, scene_attrs, obj_ids, target_ids, target_captions = zip(*batch)
    batch_scene_feat, batch_scene_attr, batch_scene_mask = process_batch_data(scene_feats, scene_attrs)
    target_ids = torch.tensor(target_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_attr": batch_scene_attr,
        "scene_mask": batch_scene_mask,
        "obj_id": obj_ids,
        "target_id": target_ids,
        "target_captions": target_captions,
        # "ids": index
    }

