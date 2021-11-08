# import torch
from algorithms import initialize_algorithm
from train import train,evaluate
import sys
import os
from utils import Logger, log_config, set_seed, log_dataset_info, get_savepath_dir
import argparser
package_directory = os.path.dirname(os.path.abspath(__file__))
TLiDB_FOLDER = os.path.join(package_directory, "..")
sys.path.append(TLiDB_FOLDER)

# TLiDB imports
from TLiDB.datasets.get_dataset import get_dataset
from TLiDB.data_loaders.data_loaders import get_train_loader

def main(config):

    save_path_dir = get_savepath_dir(config)

    # Initialize logs
    if os.path.exists(save_path_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(save_path_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    logger = Logger(os.path.join(save_path_dir, 'log.txt'), mode)

    # log configuration
    log_config(config,logger)
    set_seed(config.seed)

    # datasets dict will contain all information about the datasets: dataset name, splits, data loaders, loss function, etc.
    datasets = {split: {"datasets": [], "loaders": [], "losses": []} for split in ['train', 'dev', 'test']}

    # load data into datasets dict
    for t, d, l in zip(config.train_tasks, config.train_datasets, config.loss_functions):
        # for debugging purposes, use original data splits
        split = "train"

        cur_dataset = get_dataset(dataset=d, task=t, output_type=config.output_type,split=split)
        if config.frac < 1.0:
            cur_dataset.random_subsample(config.frac)
        datasets['train']['datasets'].append(cur_dataset)
        datasets['train']['loaders'].append(get_train_loader(cur_dataset, config.gpu_batch_size, collate_fn=cur_dataset.collate))
        datasets['train']['losses'].append(l)

        split = "dev"
        cur_dataset = get_dataset(dataset=d, task=t, output_type=config.output_type,split=split)
        if config.frac < 1.0:
            cur_dataset.random_subsample(config.frac)
        datasets['dev']['datasets'].append(cur_dataset)
        datasets['dev']['loaders'].append(get_train_loader(cur_dataset, config.gpu_batch_size, collate_fn=cur_dataset.collate))
        datasets['dev']['losses'].append(l)


    # log dataset info
    log_dataset_info(datasets, logger)

    # initialize algorithm
    algorithm = initialize_algorithm(config, datasets)

    # train
    best_val_metric = 0
    train(algorithm, datasets, config, logger, best_val_metric)

    # TODO: test, essentially just evaluate, but with test data


if __name__ == "__main__":
    config = argparser.parse_args()
    main(config)
