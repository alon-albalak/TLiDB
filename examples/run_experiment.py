# import torch
from algorithms import initialize_algorithm
from train import train,evaluate
import logging
import sys
import os
package_directory = os.path.dirname(os.path.abspath(__file__))
TLiDB_FOLDER = os.path.join(package_directory, "..")
sys.path.append(TLiDB_FOLDER)

# TLiDB imports
from TLiDB.utils import TLiDB_datasets, utils, argparser
from TLiDB.data_loaders.data_loaders import get_train_loader

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(config):
    # train, evaluate, and test a model
    if config.seed != -1:
        utils.set_seed(config.seed)

    # datasets dict will contain all information about the datasets: dataset name, splits, data loaders, loss function, etc.
    datasets = {split: {"datasets": [], "loaders": [], "losses": []} for split in ['train', 'dev', 'test']}

    # load data into datasets dict
    for t, d, l in zip(config.train_tasks, config.train_datasets, config.loss_functions):
        cur_dataset = TLiDB_datasets.DATASETS_INFO[d]['dataset_class'](task=t, output_type=config.output_type)
        if config.frac < 1.0:
            cur_dataset.random_subsample(config.frac)
        datasets['train']['datasets'].append(cur_dataset)
        datasets['train']['loaders'].append(get_train_loader(cur_dataset, config.gpu_batch_size, collate_fn=cur_dataset.collate))
        datasets['train']['losses'].append(l)

    # initialize algorithm
    algorithm = initialize_algorithm(config, datasets)

    # train
    train(algorithm, datasets, config)


if __name__ == "__main__":
    config = argparser.parse_args()
    main(config)
