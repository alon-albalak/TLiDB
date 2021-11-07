from torch.utils import data
from tqdm import tqdm
import torch
from examples.utils import detach_and_clone, collate_list, concat_t_d
from TLiDB.data_loaders.data_loaders import TLiDB_DataLoader

import logging
logger = logging.getLogger(__name__)

def run_epoch(algorithm, datasets, epoch, config, train):
    """
    Run one epoch of training or validation.
    Args:
        algorithm: (Algorithm) the algorithm to run
        datasets: (dict) contains all information about the datasets: splits, losses, etc.
        epoch: (int) the number of the epoch
        config: (Config) the configuration
        train: (boolean) True for training, False for validation (in val mode).
    """
    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)
    
    # convert all datasets into a single multi-task and multi-domain dataloader
    dataloader = TLiDB_DataLoader(datasets)

    epoch_y_true = {concat_t_d(d.task,d.dataset_name): [] for d in datasets['datasets']}
    epoch_y_pred = {concat_t_d(d.task,d.dataset_name): [] for d in datasets['datasets']}
    epoch_metadata = {concat_t_d(d.task,d.dataset_name): [] for d in datasets['datasets']}

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment the step
    pbar = tqdm(dataloader) if config.progress_bar else dataloader
    # cumulative loss during the epoch
    total_loss = {concat_t_d(d.task,d.dataset_name): 0 for d in datasets['datasets']}
    step = 0
    for batch in pbar:
        _, _, batch_metadata = batch
        batch_t_d = concat_t_d(batch_metadata['task'],batch_metadata['dataset_name'])
        if train:
            batch_results = algorithm.update(batch)
        else:
            batch_results = algorithm.evaluate(batch)
        
        # These should already be detached, but in some versions they won't get garbage
        #   collected properly if not detached again
        epoch_y_true[batch_t_d].append(detach_and_clone(batch_results['y_true']))
        epoch_y_pred[batch_t_d].append(detach_and_clone(batch_results['y_pred']))
        epoch_metadata[batch_t_d].append(detach_and_clone(batch_results['metadata']))

        total_loss[batch_t_d] += detach_and_clone(batch_results['objective']['loss_value'])
        desc = "Train losses" if train else "Validation losses"
        for t_d in total_loss:
            desc += f" | {t_d}: {total_loss[t_d]/(step+1):0.4f}"
        #     pbar.set_description(f"{desc} loss: {total_loss[t_d]:.3f}")
        # desc += f": {total_loss/(step+1):0.4f}"
        pbar.set_description(desc)
        step += 1

    epoch_y_true[batch_t_d] = collate_list(epoch_y_true[batch_t_d])
    epoch_y_pred[batch_t_d] = collate_list(epoch_y_pred[batch_t_d])


    results = {}
    for d in datasets['datasets']:
        t_d = concat_t_d(d.task,d.dataset_name)
        r, r_str = d.eval(epoch_y_true[t_d], epoch_y_pred[t_d])
        results[t_d] = r
        logger.info(r_str)
    results['epoch'] = epoch

    return results, epoch_y_pred


def train(algorithm, datasets, config):
    for epoch in range(config.num_epochs):
        # train
        epoch_result = run_epoch(algorithm, datasets['train'], epoch, config, train=True)

        # evaluate on validation set
        val_results, y_pred = run_epoch(algorithm, datasets['dev'], epoch, config, train=False)

        

def evaluate():
    pass
