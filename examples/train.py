import logging
import sys
import os
package_directory = os.path.dirname(os.path.abspath(__file__))
TLiDB_FOLDER=os.path.join(package_directory, "..")
sys.path.append(TLiDB_FOLDER)

# TLiDB imports
from TLiDB.utils import datasets, metrics, utils
from TLiDB.data_loaders.data_loaders import get_train_loader

# models
from models.initializer import initialize_model

# utils
from examples.utils import move_to

# general ML/NLP imports
import torch

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

    # load data
    dataset = datasets.DATASETS_INFO[config.dataset_name]['dataset_class'](task=config.task)
    dataloader = get_train_loader(dataset, config.gpu_batch_size, collate_fn=dataset.collate)

    # load model
    model=initialize_model(config, dataset)
    model.to(config.device)


    for e in range(config.num_epochs):
        for batch in dataloader:
            X, y = batch
            X = model.transform_inputs(X)
            y = model.transform_outputs(y)

            X = move_to(X, config.device)
            y = move_to(y, config.device)

            logits = model(X)
            loss = model.loss(logits, y)




if __name__=="__main__":
    config = utils.parse_args()
    main(config)
