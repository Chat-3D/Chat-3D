import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob

logger = logging.getLogger(__name__)


class PTBaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.feats = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        scene_id = self.anno[index]["scene_id"]
        obj_id = self.anno[index]["obj_id"]
        scene_attr = self.attributes[scene_id]
        obj_num = len(scene_attr["locs"])
        scene_locs = torch.tensor(scene_attr["locs"])
        scene_colors = torch.tensor(scene_attr["colors"])
        obj_ids = scene_attr["obj_ids"] if "obj_ids" in scene_attr else [_i for _i in range(obj_num)]
        scene_attr = torch.cat([scene_locs, scene_colors], dim=1)
        scene_feat = []
        target_id = 0
        for _i, _id in enumerate(obj_ids):
            if _id == obj_id:
                target_id = _i
            item_id = "_".join([scene_id, f"{_id:02}"])
            scene_feat.append(self.feats[item_id])
        scene_feat = torch.stack(scene_feat, dim=0)
        return scene_id, obj_id, target_id, scene_feat, scene_attr


def process_batch_data(scene_feats, scene_attrs):
    max_obj_num = max([e.shape[0] for e in scene_feats])
    # max_obj_num = 110
    batch_size = len(scene_feats)
    batch_scene_feat = torch.zeros(batch_size, max_obj_num, scene_feats[0].shape[-1])
    batch_scene_attr = torch.zeros(batch_size, max_obj_num, scene_attrs[0].shape[-1])
    batch_scene_mask = torch.zeros(batch_size, max_obj_num, dtype=torch.long)
    for i in range(batch_size):
        batch_scene_feat[i][:scene_feats[i].shape[0]] = scene_feats[i]
        batch_scene_attr[i][:scene_attrs[i].shape[0]] = scene_attrs[i]
        batch_scene_mask[i][:scene_feats[i].shape[0]] = 1
    return batch_scene_feat, batch_scene_attr, batch_scene_mask