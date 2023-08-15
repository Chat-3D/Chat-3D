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
        self.data_root, self.attribute_file, self.captions_noun_file = ann_file[:3]

        logger.info('Load json file')
        self.attributes = json.load(open(self.attribute_file, 'r'))
        self.captions_noun = json.load(open(self.captions_noun_file, 'r'))
        annos = []
        for k, v in self.captions_noun.items():
            tmp = k.split("_")
            obj_id = int(tmp[-1])
            scene_id = "_".join(tmp[:-1])
            target_captions = []
            for caption in v:
                if len(caption) == 0:
                    continue
                eid = caption.find(".")
                if eid != -1:
                    caption = caption[:eid]
                caption = caption.replace(",", "").replace("(", "").replace(")", "")
                target_captions.append(caption)
            annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "captions": target_captions
            })
        self.anno = annos

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, obj_id, scene_feat, scene_attr = self.get_anno(index)
        target_captions = self.anno[index]["captions"]
        return scene_feat, scene_attr, obj_id, target_captions


def s1_collate_fn(batch):
    scene_feats, scene_attrs, obj_ids, target_captions = zip(*batch)
    batch_scene_feat, batch_scene_attr, batch_scene_mask = process_batch_data(scene_feats, scene_attrs)
    obj_ids = torch.tensor(obj_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_attr": batch_scene_attr,
        "scene_mask": batch_scene_mask,
        "target_id": obj_ids,
        "target_captions": target_captions,
        # "ids": index
    }

