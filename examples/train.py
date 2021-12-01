from torch.utils import data
from tqdm import tqdm
import torch
from examples.utils import detach_and_clone, collate_list, concat_t_d, save_algorithm_if_needed
from TLiDB.data_loaders.data_loaders import TLiDB_DataLoader

def run_epoch(algorithm, datasets, epoch, config, logger, train):
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
    task_datasets = [concat_t_d(d.task, d.dataset_name) for d in datasets['datasets']]

    epoch_y_true = {t_d: [] for t_d in task_datasets}
    epoch_y_pred = {t_d: [] for t_d in task_datasets}
    # TODO: unclear whether epoch metadata is useful
    # epoch_metadata = {t_d: [] for t_d in task_datasets}

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment the step
    pbar = tqdm(dataloader) if config.progress_bar else dataloader
    # cumulative loss during the epoch
    total_loss = {t_d: 0 for t_d in task_datasets}
    step = {t_d: 0 for t_d in task_datasets}
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
        y_pred = detach_and_clone(batch_results['y_pred'])
        
        epoch_y_pred[batch_t_d].append(y_pred)
        # TODO: unclear whether epoch metadata is useful
        # epoch_metadata[batch_t_d].append(detach_and_clone(batch_results['metadata']))

        total_loss[batch_t_d] += detach_and_clone(batch_results['objective']['loss_value'])
        desc = "Train losses" if train else "Validation losses"
        for t_d in task_datasets:
            desc += f" | {t_d}: {total_loss[t_d]/(step[t_d]+1):0.4f}"
        #     pbar.set_description(f"{desc} loss: {total_loss[t_d]:.3f}")
        # desc += f": {total_loss/(step+1):0.4f}"
        pbar.set_description(desc)
        step[batch_t_d] += 1

        # TODO: Option for 'log every n steps'

    for t_d in task_datasets:
        epoch_y_true[t_d] = collate_list(epoch_y_true[t_d])
        epoch_y_pred[t_d] = collate_list(epoch_y_pred[t_d])


    results = {}
    logger.write('Epoch eval:\n')
    for d in datasets['datasets']:
        t_d = concat_t_d(d.task,d.dataset_name)
        # TODO: Alon left off here
        # need to set up the evaluation so that Seq2Seq models work
        # at the moment only EncoderAlgorithms work
        r, r_str = d.eval(epoch_y_pred[t_d], epoch_y_true[t_d])
        results[t_d] = r
        logger.write(f"{d.dataset_name} {d.task}-\n{r_str}\n")

    return results, epoch_y_pred


def train(algorithm, datasets, config, logger, best_val_metric):
    for epoch in range(config.num_epochs):
        logger.write(f'\nEpoch {epoch}\n')
        # train
        epoch_result = run_epoch(algorithm, datasets['train'], epoch, config, logger, train=True)

        # evaluate on validation set
        val_results, y_pred = run_epoch(algorithm, datasets['dev'], epoch, config, logger, train=False)
        val_metrics = [val_results[d][m] for d in val_results for m in val_results[d]]
        cur_val_metric = sum(val_metrics)/len(val_metrics)
        logger.write(f'Validation metric: {cur_val_metric:0.4f}\n')

        if best_val_metric is None:
            is_best=True
        else:
            is_best = cur_val_metric > best_val_metric

        if is_best:
            best_val_metric = cur_val_metric
            logger.write(f'Epoch {epoch} gives best validation result so far.\n')

        # save algorithm and model
        save_algorithm_if_needed(algorithm, epoch, config, best_val_metric, is_best)

        # TODO: save predictions if we want

        logger.write('\n')
        logger.flush()


def evaluate():
    pass