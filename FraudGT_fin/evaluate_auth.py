import copy
import logging
import time
from tqdm import tqdm
import datetime, os, logging 
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import mask_to_index, index_to_mask
from fraudGT.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_loader, get_loader
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.register import register_train
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.optimizer import create_optimizer, create_scheduler
from fraudGT.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from fraudGT.graphgym.utils.comp_budget import params_count

from fraudGT.utils import cfg_to_dict, flatten_dict, make_wandb_name
from fraudGT.utils import (new_optimizer_config, new_scheduler_config)
from fraudGT.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats
from fraudGT.train.custom_train import eval_epoch


from fraudGT.graphgym.cmd_args import parse_args
from fraudGT.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from fraudGT.graphgym.loader import create_loader
from fraudGT.graphgym.logger import setup_printing
from fraudGT.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.train import train
from fraudGT.graphgym.utils.comp_budget import params_count
from fraudGT.graphgym.utils.device import auto_select_device
from fraudGT.graphgym.register import train_dict
from torch_geometric import seed_everything

from fraudGT.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from fraudGT.logger import create_logger
from fraudGT.utils import (new_optimizer_config, new_scheduler_config, \
                             custom_set_out_dir, custom_set_run_dir)

from fraudGT.datasets.aml_dataset import AMLDataset



def run():
    global cfg
    def run_loop_settings():
        """Create main loop execution settings based on the current cfg.

        Configures the main execution loop to run in one of two modes:
        1. 'multi-seed' - Reproduces default behaviour of GraphGym when
            args.repeats controls how many times the experiment run is repeated.
            Each iteration is executed with a random seed set to an increment from
            the previous one, starting at initial cfg.seed.
        2. 'multi-split' - Executes the experiment run over multiple dataset splits,
            these can be multiple CV splits or multiple standard splits. The random
            seed is reset to the initial cfg.seed value for each run iteration.

        Returns:
            List of run IDs for each loop iteration
            List of rng seeds to loop over
            List of dataset split indices to loop over
        """
        if len(cfg.run_multiple_splits) == 0:
            # 'multi-seed' run mode
            num_iterations = args.repeat
            seeds = [cfg.seed + x for x in range(num_iterations)]
            split_indices = [cfg.dataset.split_index] * num_iterations
            run_ids = seeds
        else:
            # 'multi-split' run mode
            if args.repeat != 1:
                raise NotImplementedError("Running multiple repeats of multiple "
                                        "splits in one run is not supported.")
            num_iterations = len(cfg.run_multiple_splits)
            seeds = [cfg.seed] * num_iterations
            split_indices = cfg.run_multiple_splits
            run_ids = split_indices
        return run_ids, seeds, split_indices

    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    cfg.pretrained.reset_prediction_head = False



    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        setup_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        if args.gpu == -1:
            auto_select_device(strategy='greedy')
        else:
            logging.info('Select GPU {}'.format(args.gpu))
            if cfg.device == 'auto':
                cfg.device = 'cuda:{}'.format(args.gpu)
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline

        # cfg.train.iter_per_epoch = 10000
        # cfg.val.iter_per_epoch = 10000
        # cfg.train.batch_size = 5000
     #   cfg.test.iter_per_epoch = 7000


        #Ændre denne til ikke være udkommenteret når evaluation på alt køres
        cfg.train.sampler = "link_neighbor_all"
        cfg.val.sampler = "link_neighbor_all"


     #måske øge antal batches i hver epoch? 
        seed_everything(28)

        loaders, dataset = create_loader(returnDataset=True)
        loggers = create_logger()
        seed_everything(42)
        model = create_model(dataset=dataset)
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
             )
        # optimizer = create_optimizer(model.named_parameters(), #model.named_parameters(),
        #                               new_optimizer_config(cfg))
        # scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # print(scheduler)
    #     # Print model info
      #  logging.info(model)
     ##   logging.info(cfg)
        cfg.params = params_count(model)
     #   print(model)
     #   logging.info('Num parameters: %s', cfg.params)
    #    print('Model head is: \n\n',model.post_gt, flush =True)

        # print("Total dataset size:", len(dataset['train']))

        print('loader train', len(loaders[0]), flush=True)
        print('loader val', len(loaders[1]), flush=True)
        print('loader test', len(loaders[2]), flush=True)

        # loaders[0].set_step(10000)

        # cfg.train.iter_per_epoch = 10000 #Står to steder
        # cfg.val.iter_per_epoch = 10000 #Står to steder


        eval_epoch(loggers[0], loaders[0], model, split='train', return_pred=True, only_test=False)
        
        eval_epoch(loggers[1], loaders[1], model, split='val', return_pred=True, only_test=False)

        eval_epoch(loggers[2], loaders[2], model, split='test', return_pred=True, only_test=False)

        # loggers[0].write_epoch(1)
        # loggers[1].write_epoch(1)
        # loggers[2].write_epoch(1)



if __name__ == '__main__':
    run()