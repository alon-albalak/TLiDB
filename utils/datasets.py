import os
import json
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import logging
from utils.utils import DATASETS_INFO
logger = logging.getLogger(__name__)

# loads dataset into memory, caches them if they are not already on disk

def download_and_unzip(url, extract_to='.'):
    logger.info(f"Waiting for response from {url}")
    http_response = urlopen(url)
    logger.info("Downloading data from {url}")
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def load_dataset_local(name):
    ds = {}
    for root, dirs, files in os.walk(f"datasets/TLiDB_{name}"):
        for file in files:
            if file.endswith(".json") and file!="sample_format.json":
                ds[file[:-5]] = json.load(open(f"{root}/{file}"))
    return ds

def load_dataset(name):
    assert(name in DATASETS_INFO.keys()), f"Could not find {name} in the datasets\
                \nTry using any of {' '.join(list(DATASETS_INFO.keys()))}"

    # download and unzip dataset if needed
    if f"TLiDB_{name}" not in os.listdir("datasets"):
        download_and_unzip(DATASETS_INFO[name]['url'],"datasets")
        logger.info(f"Extracted files to /datasets/{name}")

    ds = load_dataset_local(name)

    return ds


class Dataset():
    """ Children datasets must implement the get_data function
    """
    def __init__(self, dataset_name, task):
        self.dataset = load_dataset(dataset_name)

        # temporary way to deal with undefined dataset splits
        random_subset = list(self.dataset.keys())[0]
        self.dataset = self.dataset[random_subset]

        assert task in self.dataset['metadata']['tasks']

    def get_data():
        pass

class clinc150_dataset(Dataset):
    def get_data(tokenizer):
        pass