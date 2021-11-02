# import torch
from algorithms.initializer import initialize_algorithm
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

    datasets = {split: {} for split in ['train', 'dev', 'test']}

    # load data
    # TODO: add support for multiple datasets
    datasets['train']['dataset'] = TLiDB_datasets.DATASETS_INFO[config.dataset_name]['dataset_class'](task=config.task, output_type=config.output_type)
    if config.frac < 1.0:
        datasets['train']['dataset'].random_subsample(config.frac)
    datasets['train']['loader'] = get_train_loader(datasets['train']['dataset'], config.gpu_batch_size, collate_fn=datasets['train']['dataset'].collate)

    # initialize algorithm
    algorithm = initialize_algorithm(config, datasets)

    train(algorithm, datasets, config)


if __name__ == "__main__":
    config = argparser.parse_args()
    main(config)
