from tqdm import tqdm
import torch
from examples.utils import detach_and_clone, collate_list
from TLiDB.data_loaders.data_loaders import TLiDB_DataLoader

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
    
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    # convert all datasets into a single multi-task and multi-domain dataloader
    dataloader = TLiDB_DataLoader(datasets)

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment the step
    pbar = tqdm(dataloader) if config.progress_bar else dataloader
    total_loss = 0 # cumulative loss during the epoch
    step = 0
    for batch in pbar:
        if train:
            batch_results = algorithm.update(batch)
        else:
            batch_results = algorithm.evaluate(batch)
        
        # These should already be detached, but in some versions they won't get garbage
        #   collected properly if not detached again
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        epoch_y_pred.append(detach_and_clone(batch_results['y_pred']))
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        total_loss += detach_and_clone(batch_results['objective']['loss_value'])
        desc = "Train" if train else "Validation"
        desc += f": {total_loss/(step+1):0.4f}"
        pbar.set_description(desc)
        step += 1

    epoch_y_true = collate_list(epoch_y_true)
    epoch_y_pred = collate_list(epoch_y_pred)

def train(algorithm, datasets, config):
    for epoch in range(config.num_epochs):
        run_epoch(algorithm, datasets['train'], epoch, config, train=True)

def evaluate():
    pass
