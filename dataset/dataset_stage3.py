import logging
import os
import json
import sqlite3
import random
from os.path import basename

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data
import glob

logger = logging.getLogger(__name__)


class S3PTDataset(PTBaseDataset):

    def __init__(self, ann_file, system="", **kwargs):
        super().__init__()
        self.data_root, self.attribute_file, self.conv_file = ann_file[:3]

        self.system = system
        self.role = ("Human", "Assistant")
        self.pc_token = "<Target><TargetHere></Target>"
        self.scene_token = "<Scene><SceneHere></Scene>"
        self.begin_signal = "###"
        self.end_signal = " "

        logger.info('Load json file')
        self.attributes = json.load(open(self.attribute_file, 'r'))
        self.convs = json.load(open(self.conv_file, 'r'))
        annos = []
        for k, v in self.convs.items():
            if len(v) == 0:
                continue
            tmp = k.split("_")
            obj_id = int(tmp[-1])
            scene_id = "_".join(tmp[:-1])
            annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "QA": v
            })
        self.anno = annos

    def __len__(self):
        return len(self.anno)

    def process_qa(self, qas, msg=""):
        conversation = self.system + self.end_signal
        # conversation += (
        #     # self.begin_signal + self.role[0] + ": " +
        #     " " + self.pc_token + " " + self.scene_token + msg.rstrip() + self.end_signal
        # )
        for idx, qa in enumerate(qas):
            q = qa["Question"]
            a = qa["Answer"]
            a = a.replace("\n\n", " ")
            conversation += (self.begin_signal + self.role[0] + ": " + q + self.end_signal)
            conversation += (self.begin_signal + self.role[1] + ": " + a + self.end_signal)
        conversation += self.begin_signal
        return conversation

    def __getitem__(self, index):
        scene_id, obj_id, scene_feat, scene_attr = self.get_anno(index)
        conversation = self.process_qa(self.anno[index]["QA"])
        return scene_feat, scene_attr, obj_id, conversation


def s3_collate_fn(batch):
    scene_feats, scene_attrs, obj_ids, conversations = zip(*batch)
    batch_scene_feat, batch_scene_attr, batch_scene_mask = process_batch_data(scene_feats, scene_attrs)
    obj_ids = torch.tensor(obj_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_attr": batch_scene_attr,
        "scene_mask": batch_scene_mask,
        "target_id": obj_ids,
        "conversations": conversations
        # "ids": index
    }
