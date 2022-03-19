from tqdm import tqdm
import torch
from examples.utils import detach_and_clone, collate_list, concat_t_d, save_algorithm_if_needed, save_pred_if_needed
from TLiDB.data_loaders.data_loaders import TLiDB_DataLoader

def run_epoch(algorithm, datasets, config, logger, train):
    """
    Run one epoch of training or validation.
    Args:
        algorithm: (Algorithm) the algorithm to run
        datasets: (dict) contains all information about the datasets: splits, losses, etc.
        config: (Config) the configuration
        logger: (Logger) the logger
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
            batch_results = algorithm.update(batch,sum(step.values()))
        else:
            batch_results = algorithm.evaluate(batch)
        
        # These should already be detached, but in some versions they won't get garbage
        #   collected properly if not detached again
        epoch_y_true[batch_t_d].append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        
        epoch_y_pred[batch_t_d].append(y_pred)

        total_loss[batch_t_d] += detach_and_clone(batch_results['objective']['loss_value'])
        desc = "Train losses" if train else "Validation losses"
        for t_d in task_datasets:
            desc += f" | {t_d}: {total_loss[t_d]/(step[t_d]+1):0.4f}"

        pbar.set_description(desc)
        step[batch_t_d] += 1

        # TODO: Option for 'log every n steps'

    for t_d in task_datasets:
        epoch_y_true[t_d] = collate_list(epoch_y_true[t_d])
        epoch_y_pred[t_d] = collate_list(epoch_y_pred[t_d])


    # This loop is determined by the model/task/mode(train/val)
    results = {}
    if algorithm.requires_metric_calculation():
        logger.write('Epoch eval:\n')
        for m, d in zip(datasets['metrics'],datasets['datasets']):
            t_d = concat_t_d(d.task,d.dataset_name)
            r, r_str = m.compute(epoch_y_pred[t_d], epoch_y_true[t_d])
            results[t_d] = r
            logger.write(f"{d.dataset_name} {d.task}-\n{r_str}\n")

    return results, epoch_y_pred


def train(algorithm, datasets, config, logger, epoch_offset, best_val_metric):
    for epoch in range(epoch_offset, config.num_epochs):
        logger.write(f'\nEpoch {epoch}\n')
        # train
        run_epoch(algorithm, datasets['train'], config, logger, train=True)

        # allow for training without dev set, will not save model
        if not datasets['dev'].get('datasets', None):
            continue

        # evaluate on validation set
        val_results, y_pred = run_epoch(algorithm, datasets['dev'], config, logger, train=False)
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
        save_algorithm_if_needed(algorithm, epoch, config, best_val_metric, is_best, logger)
        # save predictions
        save_pred_if_needed(y_pred, epoch, config, is_best, config.save_path_dir)

        logger.write('\n')
        logger.flush()


def evaluate(algorithm, datasets, config, logger, epoch, is_best):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split in datasets:
        for dataset, loader, metric in zip(datasets[split]['datasets'], datasets[split]['loaders'], datasets[split]['metrics']):
            epoch_y_true = []
            epoch_y_pred = []

            pbar = tqdm(iter(loader)) if config.progress_bar else iter(loader)

            for batch in pbar:
                # add batch metadata to the batch
                X, y, batch_metadata = batch
                batch_metadata['task'] = dataset.task
                batch_metadata['dataset_name'] = dataset.dataset_name
                batch_metadata['task_metadata'] = dataset.task_metadata
                batch = (X, y, batch_metadata)

                batch_results = algorithm.evaluate(batch)
                epoch_y_true.append(detach_and_clone(batch_results['y_true']))
                y_pred = detach_and_clone(batch_results['y_pred'])
                epoch_y_pred.append(y_pred)

            epoch_y_pred = collate_list(epoch_y_pred)
            epoch_y_true = collate_list(epoch_y_true)

            r, r_str = metric.compute(epoch_y_pred, epoch_y_true)
            r['epoch'] = epoch
            logger.write(f"Eval on {split} split at epoch {epoch}: {dataset.dataset_name} {dataset.task}-\n{r_str}\n")

            # skip saving train data as the dataloader will shuffle data
            if split != "train":
                save_pred_if_needed(y_pred, epoch, config, is_best, config.save_path_dir)
