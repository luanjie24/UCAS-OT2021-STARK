from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
# from pairwise import Pairwise

import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from lib.train.base_functions import *
# network related
from lib.models.stark import build_starks, build_starkst
# forward propagation related
from lib.train.actors import STARKSActor, STARKSTActor
# for import modules
import importlib
import lib.train.admin.settings as ws_settings

 

if __name__ == '__main__':

    # ====================================================导入模型配置
    script_name="stark_s"
    config_name="baseline_got10k_only"
    save_dir=".\demo"
    use_lmdb=False
    local_rank=-1 # 是否多卡
    # script_name_prv=
    # config_name_prv="baseline"

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)
    # if script_name_prv is not None and config_name_prv is not None:
    #     settings.project_path_prv = 'train/{}/{}'.format(script_name_prv, config_name_prv)

    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.use_lmdb = use_lmdb

    # 拿到当前路径
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))

    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'


    # update the default configs with config file
    # 判断yaml配置文件是否存在，yaml为之前实验后发现可能对训练有益处的参数
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    # 根据模型名载入配置，放到cfg中
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    # 将taml配置文件中的配置更新到cfg以覆盖lib.config中的默认配置
    config_module.update_config_from_file(settings.cfg_file)



    # update settings based on cfg
    update_settings(settings, cfg)


    # 训练日志
    log_dir = os.path.join(settings.save_dir, 'logs')
    # if settings.local_rank in [-1, 0]:
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))


    # ====================================================导入模型

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "stark_s":
        net = build_starks(cfg)
    elif settings.script_name == "stark_st1" or settings.script_name == "stark_st2":
        net = build_starkst(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()

    settings.device = torch.device("cuda:0")



    # ====================================================损失、优化器、超参数等的设置

    # Loss functions and Actors 主要就是一些损失的设置
    if settings.script_name == "stark_s" or settings.script_name == "stark_st1":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "stark_st2":
        objective = {'cls': BCEWithLogitsLoss()}
        loss_weight = {'cls': 1.0}
        actor = STARKSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")

    if cfg.TRAIN.DEEP_SUPERVISION:
        raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # LTRTrainer封装了训练模型的过程
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # train process
    if settings.script_name == "stark_st2":
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=True)
    else:

        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
