from tqdm import tqdm
import torch
from examples.utils import detach_and_clone, collate_list

def run_epoch(algorithm, dataset, epoch, config, train):
    """
    Run one epoch of training or validation.
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

    pbar = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
    total_loss = 0
    step = 0
    for batch in pbar:
        if train:
            batch_results = algorithm.update(batch)
        else:
            batch_results = algorithm.evaluate(batch)
        
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        epoch_y_pred.append(detach_and_clone(batch_results['y_pred']))
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        total_loss += detach_and_clone(batch_results['objective'])
        desc = "Train" if train else "Validation"
        desc += f": {total_loss/(step+1):0.4f}"
        pbar.set_description(desc)
        step += 1

    epoch_y_true = collate_list(epoch_y_true)
    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_metadata = collate_list(epoch_metadata)

def train(algorithm, datasets, config):
    for epoch in range(config.num_epochs):
        run_epoch(algorithm, datasets['train'], epoch, config, train=True)

def evaluate():
    pass
