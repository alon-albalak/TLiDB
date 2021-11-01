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


# models

# utils

# general ML/NLP imports

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
    datasets['train']['loader'] = get_train_loader(datasets['train']['dataset'], config.gpu_batch_size, collate_fn=datasets['train']['dataset'].collate)

    # initialize algorithm
    algorithm = initialize_algorithm(config, datasets)

    train(algorithm, datasets, config)


    # # load model
    # model = initialize_model(config, datasets)
    # model.to(config.device)

    # for e in range(config.num_epochs):
    #     for batch in dataloader:
    #         X, y, metadata = batch
    #         X = model.transform_inputs(X)
    #         y = model.transform_outputs(y)

    #         X = move_to(X, config.device)
    #         y = move_to(y, config.device)

    #         logits = model(X)
    #         loss = model.loss(logits, y)


if __name__ == "__main__":
    config = argparser.parse_args()
    main(config)
