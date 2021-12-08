import os
import sys
import random
import numpy as np
import torch

package_directory = os.path.dirname(os.path.abspath(__file__))
TLiDB_FOLDER = os.path.join(package_directory, "..")
sys.path.append(TLiDB_FOLDER)

# TLiDB imports
from TLiDB.datasets.get_dataset import get_dataset
from TLiDB.data_loaders.data_loaders import get_train_loader, get_eval_loader
from TLiDB.metrics.initializer import get_metric_computer

def load_datasets_split(split, config):
    split_datasets = {"datasets":[], "loaders":[], "metrics":[]}
    tasks = getattr(config, f"{split}_tasks")
    datasets = getattr(config, f"{split}_datasets")
    split = "train" if split == "finetune_train" else split
    split = "dev" if split == "finetune_dev" else split
    for t, d in zip(tasks, datasets):
        cur_dataset = get_dataset(dataset=d,task=t,model_type=config.model_type,split=split)
        if config.frac < 1.0:
            cur_dataset.random_subsample(config.frac)
        split_datasets["datasets"].append(cur_dataset)
        split_datasets["loaders"].append(get_train_loader(cur_dataset, config.gpu_batch_size, collate_fn=cur_dataset.collate))
        split_datasets["metrics"].append(get_metric_computer(cur_dataset.metrics, **cur_dataset.metric_kwargs))
    return split_datasets

def concat_t_d(task, dataset_name):
    return f"{task}_{dataset_name}"

def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        # Assume obj is a Tensor or other type
        return obj.to(device)

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")


def collate_list(inputs):
    """
    If inputs is a list of Tensors, it concatenates them all along the first dimension.

    If inputs is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If inputs is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(inputs, list):
        raise TypeError("collate_list must take in a list")
    elem = inputs[0]
    if torch.is_tensor(elem):
        return torch.cat(inputs)
    elif isinstance(elem, list):
        return [obj for sublist in inputs for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in inputs]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")


def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def get_savepath_dir(datasets, tasks, seed, log_dir, model):
    prefix = "PRETRAINED_"
    for dataset,task in zip(datasets, tasks):
        prefix += f"{dataset}.{task}_"
    if seed > -1:
        prefix += f"seed.{seed}_"
    if prefix == "PRETRAINED_":
        raise ValueError("Cannot create dir with empty name")

    prefix = os.path.join(log_dir, prefix[:-1], model)
    return prefix

def append_to_save_path_dir(save_path_dir, datasets, tasks, seed):
    postfix = "FINETUNED_"
    for dataset,task in zip(datasets, tasks):
        postfix += f"{dataset}.{task}_"
    if seed > -1:
        postfix += f"seed.{seed}_"
    if postfix == "FINETUNED_":
        raise ValueError("Cannot create dir with empty name")
    return os.path.join(save_path_dir,postfix[:-1])

def save_algorithm(algorithm, epoch, best_val_metric, path, logger):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)
    logger.write(f"Saved model to {path}\n")

def load_algorithm(algorithm, path, logger):
    state = torch.load(path)
    algorithm.load_state_dict(state['algorithm'])
    logger.write(f"Loaded model from {path}\n")
    return state['epoch'], state['best_val_metric']

def save_algorithm_if_needed(algorithm, epoch, config, best_val_metric, is_best, logger):
    if config.save_last:
        save_algorithm(algorithm,epoch,best_val_metric,os.path.join(config.save_path_dir,"last_model.pt"),logger)
    if config.save_best and is_best:
        save_algorithm(algorithm, epoch, best_val_metric,os.path.join(config.save_path_dir,"best_model.pt"),logger)

def save_pred_if_needed(y_pred, epoch, config, is_best, force_save=False, save_path_dir=None):
    if config.save_pred:
        if not save_path_dir:
            save_path_dir = get_savepath_dir(config)
        if force_save:
            save_pred(y_pred, os.path.join(save_path_dir, f"predictions_{epoch}"))
        if (not force_save) and config.save_last:
            save_pred(y_pred, os.path.join(save_path_dir, "last_predictions"))
        if config.save_best and is_best:
            save_pred(y_pred, os.path.join(save_path_dir, "best_predictions"))

def save_pred(y_pred, path_prefix):
    # Single tensor
    if torch.is_tensor(y_pred):
        y_pred_np = y_pred.numpy()
        np.save(path_prefix+'.csv', y_pred_np)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + '.pt')
    else:
        raise TypeError("Invalid type for save_pred")

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def log_config(config, logger):
    # Simply write all configuration parameters to the logger
    logger.write('Configuration:\n')
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')
    logger.flush()


def log_dataset_info(datasets, logger):
    # Simply write all dataset details to the logger
    logger.write('Datasets:\n')
    for split in datasets:
        logger.write(f'{split} | ')
        for dataset, metrics in zip(datasets[split]['datasets'], datasets[split]['metrics']):
            logger.write(f'{dataset.dataset_name}')
            logger.write(f' - {dataset.task}')
            for metric in metrics.metrics:
                logger.write(f' - {metric.name}')
            if dataset.num_classes:
                logger.write(f' - {dataset.num_classes} classes')
            logger.write(f' - {dataset.y_size} examples | ')
        logger.write('\n')
    logger.flush()