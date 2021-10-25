import os
import json
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import logging
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

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


class TLiDB_Dataset(Dataset):
    """ Children datasets must implement the get_data function
    """
    def __init__(self, dataset_name, task):
        super().__init__()
        self.dataset = load_dataset(dataset_name)

        # temporary way to deal with undefined dataset splits
        random_subset = list(self.dataset.keys())[0]
        # self.dataset = self.dataset[random_subset]

        assert task in self.dataset[random_subset]['metadata']['tasks']

    def get_data(self):
        """
        Returns data to be fed to dataloader, 
        """
        pass

    def __getitem__(self, index):
        """
        """


class clinc150_dataset(TLiDB_Dataset):
    def get_data(self, tokenizer):
        data = []
        pass


class multiwoz22_dataset(TLiDB_Dataset):
    def get_data(self, tokenizer):
        pass


class friends_ER_dataset(TLiDB_Dataset):
    def get_data(self, tokenizer):
        pass


class friends_RC_dataset(TLiDB_Dataset):
    def get_data(self, tokenizer):
        pass


class friends_QA_dataset(TLiDB_Dataset):
    def get_data(self, tokenizer):
        pass


# dataset url can be found in the google drive, where the original link is:
#   https://drive.google.com/file/d/1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq/view?usp=sharing
# and needs to be reformatted as:
#   https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq
DATASETS_INFO = {
    "multiwoz22": {"dataset_class": multiwoz22_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1N77FmuksmFZuFVwk87rQHMzw_DqdTLP2"},
    "clinc150": {"dataset_class": clinc150_dataset,
                 "url": "https://drive.google.com/uc?export=download&id=1syuXRgT2oj5d5dAMm_b83gqnnir1vF3y"},
    "friends_ER": {"dataset_class": friends_ER_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1evEtiYj9I3-lqD8JXHpknHSkYHzq2uQB"},
    "friends_RC": {"dataset_class": friends_RC_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1jQy3dQd8exl7otgJRi-fp9Ldp9ppCzmG"},
    "friends_QA": {"dataset_class": friends_QA_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=11DELN1S722Yi4XNn_YJ0NwvJy4io8Gyi"}
}
