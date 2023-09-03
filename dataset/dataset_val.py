import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data
import glob

logger = logging.getLogger(__name__)


class ValPTDataset(PTBaseDataset):

    def __init__(self, ann_file, system_path="", stage=2, **kwargs):
        super().__init__()
        self.feat_file, self.attribute_file, self.prompt_file = ann_file[:3]
        with open(system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        self.feats = torch.load(self.feat_file)
        self.attributes = json.load(open(self.attribute_file, 'r'))
        self.anno = json.load(open(self.prompt_file, 'r'))
        if stage == 2:
            self.prompt_template = "###Human: {} ###Assistant: "
        else:
            self.prompt_template = "###Human: {} ###"

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, obj_id, target_id, scene_feat, scene_attr = self.get_anno(index)
        prompt = self.system + " " + self.prompt_template.format(self.anno[index]["prompt"])
        ref_captions = self.anno[index]["ref_captions"] if "ref_captions" in self.anno[index] else []
        qid = self.anno[index]["qid"] if "qid" in self.anno[index] else 0
        return scene_feat, scene_attr, obj_id, target_id, prompt, ref_captions, scene_id, qid


def valuate_collate_fn(batch):
    scene_feats, scene_attrs, obj_ids, target_ids, prompts, ref_captions, scene_ids, qids = zip(*batch)
    batch_scene_feat, batch_scene_attr, batch_scene_mask = process_batch_data(scene_feats, scene_attrs)
    target_ids = torch.tensor(target_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_attr": batch_scene_attr,
        "scene_mask": batch_scene_mask,
        "obj_id": obj_ids,
        "target_id": target_ids,
        "custom_prompt": prompts,
        "ref_captions": ref_captions,
        "scene_id": scene_ids,
        "qid": qids
        # "ids": index
    }

