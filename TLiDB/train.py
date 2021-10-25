import logging
import sys
from utils import datasets, metrics, utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def main(**kwargs):
    if kwargs["seed"] != -1:
        utils.set_seed(kwargs["seed"])

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'])

    # load data
    dataset = datasets.DATASETS_INFO[kwargs['dataset_name']]['dataset_class'](dataset_name=kwargs['dataset_name'], task=kwargs['task'])
    data = dataset.get_data(tokenizer=tokenizer)

    # load model
    model=AutoModelForSequenceClassification.from_pretrained(kwargs['model_name_or_path'],num_labels=len(dataset['metadata']['task_metadata'][kwargs['task']]['labels']))
    model.to(kwargs['device'])




if __name__=="__main__":
    args = utils.parse_args()
    main(**args)
