import logging
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from config.config import cfg



def weighted_cross_entropy(pred, true, epoch):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        # calculating label weights for weighted loss computation
        if cfg.model.loss_fun_weight is None:
            V = true.size(0)
            n_classes = pred.shape[1] if pred.ndim > 1 else 2
            label_count = torch.bincount(true)
            label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
            cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
            cluster_sizes[torch.unique(true)] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes > 0).float()
        else:
            weight = torch.tensor(cfg.model.loss_fun_weight, device=torch.device(cfg.device))
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)

def compute_loss(pred, true, epoch=None):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    bce_loss_no_red = nn.BCEWithLogitsLoss(reduction='none')
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    # for func in register.loss_dict.values():
    value = weighted_cross_entropy(pred, true, epoch)
    if value is not None:
        return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary or multilabel
        else:
            # num_positives = true.sum().item()
            # num_negatives = len(true) - num_positives

            # # Calculate the weight for each class
            # weight_for_0 = num_positives / len(true)# * 3
            # weight_for_1 = num_negatives / len(true)

            # # Create a tensor of weights with the same shape as your labels
            # weights = true * weight_for_1 + (1 - true) * weight_for_0
            # weights = weights.to(pred.device)

            # true = true.float()

            # loss = bce_loss_no_red(pred, true)
            # loss = (loss * weights).mean()
            # return loss, torch.sigmoid(pred)
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

def train_epoch(cur_epoch, logger, loader, model, optimizer, scheduler, batch_accumulation):
    pbar = tqdm(total=len(loader), disable=not cfg.train.tqdm)
    pbar.set_description(f'Train epoch')

    model.train()

    iterator = iter(loader)

    if cfg.model.type == 'LPModel':  # Handle label propagation specially
        # We don't need to train label propagation
        time_start = time.time()
        batch = next(iterator, None)
        batch.split = 'train'
        batch.to(torch.device('cpu'))
        
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().cpu()
        _pred = pred_score.detach().cpu()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=0,
                            dataset_name=cfg.dataset.name)
        pbar.update(1)
        return

    optimizer.zero_grad()
    it = 0
    time_start = time.time()
    while True:
        try:
            batch = next(iterator, None)
            it += 1
            if batch is None:
                break

            if isinstance(batch, Data) or isinstance(batch, HeteroData):
                batch.split = 'train'
                batch.to(torch.device('cpu'))
            else:  # NAGphormer, HINo
                batch = [x.to(torch.device('cpu')) for x in batch]

            pred, true = model(batch)
            if cfg.model.loss_fun == 'curriculum_learning_loss':
                loss, pred_score = compute_loss(pred, true, cur_epoch)
            else:
                loss, pred_score = compute_loss(pred, true)
            _true = true.detach().cpu()
            _pred = pred_score.detach().cpu()

            loss.backward()

            # Parameters update after accumulating gradients for given num. batches.
            if ((it + 1) % batch_accumulation == 0) or (it + 1 == len(loader)):
                if cfg.optim.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.optim.clip_grad_norm_value)
                optimizer.step()
                optimizer.zero_grad()
            cfg.params = params_count(model)
            logger.update_stats(true=_true,
                                pred=_pred,
                                loss=loss.detach().cpu().item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name)
            pbar.update(1)
            time_start = time.time()
        except RuntimeError as e:
            if "cannot sample n_sample <= 0 samples" in str(e):
                print(f"Skipping batch due to error: {e}")
                continue
            else:
                # If it's a different error, re-raise it
                raise


def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(cur_epoch, loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch, start_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch, start_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    logging.info('Task done, results saved in %s', cfg.run_dir)
