import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from dataset.dataset_stage1 import s1_collate_fn
from dataset.dataset_stage2 import s2_collate_fn
from dataset.dataset_stage3 import s3_collate_fn
from dataset.dataset_val import valuate_collate_fn
from models.chat3d import Chat3D
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

import aac_metrics
import numpy as np
from tqdm import tqdm

import json
import os

logger = logging.getLogger(__name__)
max_bleus = [0.] * 4

def train(
    model,
    model_without_ddp,
    train_loaders,
    val_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    eval_metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    loss_names = ["loss"]

    media_types = get_media_types(train_loaders)

    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(
                f"{m}-{name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    eval_freq = 100  # len(train_loader)

    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, batch) in enumerate(iterator):
        for k in batch.keys():
            if k in ["scene_feat", "scene_attr", "scene_mask", "target_id"]:
                batch[k] = batch[k].to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16):
            loss_dict = model(**batch)
            loss = loss_dict["loss"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if config.optimizer.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if global_step % eval_freq == 0 or i == len(train_loader)-1:
            val_metrics = evaluate(model, model_without_ddp, val_loaders, epoch, global_step, device, config,
                                              early_stop=i != len(train_loader)-1)
            if is_main_process():
                for k, v in val_metrics.items():
                    if k not in eval_metric_logger.meters:
                        eval_metric_logger.add_meter(k, SmoothedValue(window=1, fmt="{value:.4f}"))
                eval_metric_logger.update(**val_metrics)
            if is_main_process() and config.wandb.enable:
                logs = eval_metric_logger.get_global_avg_dict()
                log_dict_to_wandb(logs, step=global_step, prefix="val/")

        if config.debug and global_step % 20 == 0:
            logger.info("debug mode, break training loop")
            break

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step


def evaluate(
    model,
    model_without_ddp,
    val_loaders,
    epoch,
    global_step,
    device,
    config,
    early_stop=False
):
    model.eval()

    media_types = get_media_types(val_loaders)

    if config.distributed:
        for d in val_loaders:
            d.sampler.set_epoch(epoch)

    val_loader = MetaLoader(name2loader=dict(list(zip(media_types, val_loaders))))

    # scores = []
    sample_freq = len(val_loader) // 5
    cosine_scores = []
    preds = []
    targets = []
    save_preds = []

    for i, (media_type, batch) in tqdm(enumerate(val_loader)):
        for k in batch.keys():
            if k in ["scene_feat", "scene_attr", "scene_mask", "target_id"]:
                batch[k] = batch[k].to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16) and torch.no_grad():
            pred = model(**batch, is_eval=True)
            if "target_captions" in batch:
                cosine_scores.append(1. - pred["loss"].detach().cpu())
            if "custom_prompt" in batch:
                if len(batch["ref_captions"][0]) > 0:
                    target = batch["ref_captions"]
                    tmp_pred = [p.replace("\n", " ") for p in pred]
                    preds += tmp_pred
                    targets += target
                    if i % sample_freq == 0:
                        logger.info(f"\n[Pred] {tmp_pred[0]}\n[Target] {target[0][0]}")
                batch_size = len(pred)
                for bi in range(batch_size):
                    scene_id = batch["scene_id"][bi]
                    obj_id = int(batch["target_id"][bi].detach().cpu())
                    qid = batch["qid"][bi]
                    prompt = batch["custom_prompt"][bi]
                    tmp_pred = pred[bi]
                    save_preds.append({
                        "scene_id": scene_id,
                        "obj_id": obj_id,
                        "qid": qid,
                        "prompt": prompt,
                        "pred": tmp_pred
                    })
                if i % sample_freq == 0:
                    print(save_preds[-1])
                if early_stop:
                    break

    val_scores = {}
    logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
    if is_main_process() and len(cosine_scores) > 12:
        val_scores["cosine_sim"] = float(torch.stack(cosine_scores).mean())
        for k, v in val_scores.items():
            logger.info(f"{k}: {v}")

    if is_main_process() and len(preds) > 12:
        val_scores, _ = aac_metrics.evaluate(preds, targets)
        logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
        for k, v in val_scores.items():
            logger.info(f"{k}: {v}")

    if is_main_process() and len(save_preds) > 20:
        save_preds = sorted(save_preds, key=lambda x: f"{x['scene_id']}_{x['obj_id']:03}_{x['qid']:02}")
        with open(os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}.json"), "w") as f:
            json.dump(save_preds, f, indent=4)

    return val_scores


def setup_dataloaders(config):
    # train datasets, create a list of data loaders
    train_datasets, val_datasets = create_dataset(config)
    media_types = get_media_types(train_datasets)

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_samplers = create_sampler(
            train_datasets, [True] * len(media_types), num_tasks, global_rank
        )
        val_samplers = create_sampler(
            val_datasets, [True] * len(media_types), num_tasks, global_rank
        )

    else:
        train_samplers = [None] * len(media_types)
        val_samplers = [None] * len(media_types)

    if config.model.stage == 1:
        train_collate_fn = val_collate_fn = s1_collate_fn
        train_batch_size = val_batch_size = config.s1_batch_size
    elif config.model.stage == 2:
        train_collate_fn = s2_collate_fn
        val_collate_fn = valuate_collate_fn
        train_batch_size = val_batch_size = config.s2_batch_size
    elif config.model.stage == 3:
        train_collate_fn = s3_collate_fn
        val_collate_fn = valuate_collate_fn
        train_batch_size = val_batch_size = config.s3_batch_size
    else:
        raise NotImplementedError

    train_loaders = create_loader(
        train_datasets,
        train_samplers,
        batch_size=[train_batch_size * len(media_types)],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[train_collate_fn] * len(media_types),
    )  # [0]
    val_loaders = create_loader(
        val_datasets,
        val_samplers,
        batch_size=[val_batch_size * len(media_types)],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[val_collate_fn] * len(media_types),
    )

    return train_loaders, val_loaders, media_types


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, val_loaders, train_media_types = setup_dataloaders(
        config
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    model_cls = eval(config.model.get('model_cls', 'Chat3D'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    logger.info("Start training")
    start_time = time.time()
    if not config.evaluate:
        for epoch in range(start_epoch, config.scheduler.epochs):
            global_step = train(
                model,
                model_without_ddp,
                train_loaders,
                val_loaders,
                optimizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
            )

            if is_main_process():
                logger.info(f"Epoch {epoch}")
                param_grad_dic = {
                    k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
                }
                state_dict = model_without_ddp.state_dict()
                for k in list(state_dict.keys()):
                    if k in param_grad_dic.keys() and not param_grad_dic[k]:
                        # delete parameters that do not require gradient
                        del state_dict[k]
                save_obj = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if config.get("save_latest", False):
                    torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
                else:
                    torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

            dist.barrier()

    if config.evaluate:
        evaluate(model, model_without_ddp, val_loaders, start_epoch-1, global_step, device, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
