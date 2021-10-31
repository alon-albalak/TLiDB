import os
import json
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER=os.path.join(package_directory, "../datasets")

def download_and_unzip(url, extract_to='.'):
    logger.info(f"Waiting for response from {url}")
    http_response = urlopen(url)
    logger.info("Downloading data from {url}")
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def load_dataset_local(name, dataset_folder):
    ds = {}
    for root, dirs, files in os.walk(f"{dataset_folder}/TLiDB_{name}"):
        for file in files:
            if file.endswith(".json") and file!="sample_format.json":
                ds[file[:-5]] = json.load(open(f"{root}/{file}"))
    
    if len(ds.keys()) == 1:
        ds = ds[list(ds.keys())[0]]
    return ds

def load_dataset(name, dataset_folder):
    assert(name in DATASETS_INFO.keys()), f"Could not find {name} in the datasets\
                \nTry using any of {' '.join(list(DATASETS_INFO.keys()))}"

    # download and unzip dataset if needed
    if f"TLiDB_{name}" not in os.listdir(dataset_folder):
        download_and_unzip(DATASETS_INFO[name]['url'], dataset_folder)
        logger.info(f"Extracted files to {dataset_folder}/{name}")

    ds = load_dataset_local(name, dataset_folder)

    return ds


class TLiDB_Dataset(Dataset):
    """
    Abstract dataset class for all TLiDB datasets
    """
    def __init__(self, dataset_name, task, dataset_folder=DATASET_FOLDER):
        super().__init__()
        self.dataset = load_dataset(dataset_name, dataset_folder)
        self.task_metadata = self.dataset['metadata']['task_metadata']

        if task == "response_generation":
            #TODO:
            # gather data such that _input_array contains all utterances up to the current one
            # and _y_array contains the response
            pass
        else:
            self.metrics = self.task_metadata[task]['metrics']

    @property
    def dataset_name(self):
        return self._dataset_name
    
    @property
    def collate(self):
        """
        Returns collate function to be used with dataloader
        By default returns None -> uses default torch collate function
        """
        return getattr(self, "_collate", None)

    @property
    def y_array(self):
        """
        Targets for the model to predict, can be labels for classification or string for generation tasks
        """
        return self._y_array

    @property
    def y_size(self):
        """
        Number of elements in the target
        For standard classification and text generation, y_size = 1
        For multi-class or multi-task prediction tasks, y_size > 1
        """
        return self._y_size

    @property
    def num_classes(self):
        """
        Returns number of classes in the dataset
        """
        return getattr(self, "_num_classes", None)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset
        """
        x = self.get_input(idx)
        y = self.get_target(idx)
        return x, y

    def get_input(self, idx):
        return NotImplementedError

    def get_target(self, idx):
        return self.y_array[idx]

    def _collate(self, batch):
        return NotImplementedError

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.y_array)

    def eval(self, y_pred, y_true):
        """
        Evaluates the model performance
        Args:
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Returns:
            - results (dict): Dictionary of metrics
        """
        return NotImplementedError


class clinc150_dataset(TLiDB_Dataset):
    _dataset_name = "clinc150"
    _tasks = ["intent_detection"]
    def __init__(self, task, dataset_folder=DATASET_FOLDER):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self.dataset_name, task, dataset_folder=dataset_folder)
        # needs to initialize input_array, y_array, y_size, num_classes
        self.labels = self.task_metadata[task]['labels']
        self._num_classes = len(self.labels)
        self._input_array = []
        self._y_array = []
        for datum in self.dataset['data']:
            utterance = datum['dialogue'][0]
            self._input_array.append(utterance['utterance'])
            # y_array = [0 for _ in range(self._num_classes)]
            label = f"{utterance['intent_detection']['domain']}_{utterance['intent_detection']['intent']}"
            # y_array[self.labels.index(label)] = 1
            # self._y_array.append(y_array)
            self._y_array.append(self.labels.index(label))
        self._y_size = len(self._y_array)

    def get_input(self, idx):
        return self._input_array[idx]

    def _collate(self, batch):
        X,y = [], []
        for item in batch:
            X.append(item[0])
            y.append(item[1])
        return X,y

class multiwoz22_dataset(TLiDB_Dataset):
    _dataset_name = "multiwoz22"
    _tasks = ["dialogue_state_tracking","dialogue_response_generation"]
    def __init__(self, task, dataset_folder=DATASET_FOLDER):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self.dataset_name, task, dataset_folder=dataset_folder)

class friends_ER_dataset(TLiDB_Dataset):
    _dataset_name = "friends_ER"

class friends_RC_dataset(TLiDB_Dataset):
    _dataset_name = "friends_RC"

class friends_QA_dataset(TLiDB_Dataset):
    _dataset_name = "friends_QA"

# dataset url can be found in the google drive, where the original link is:
#   https://drive.google.com/file/d/1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq/view?usp=sharing
# and needs to be reformatted as:
#   https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq
DATASETS_INFO = {
    "multiwoz22": {"dataset_class": multiwoz22_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1ZYiKM6D2-b8HfIP_jHh6-YdtJ6sZfssQ"},
    "clinc150": {"dataset_class": clinc150_dataset,
                 "url": "https://drive.google.com/uc?export=download&id=1dG6KXQ6L7xpbnWmhW9Xo3vPSfYstk43E"},
    "friends_ER": {"dataset_class": friends_ER_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1hjbtUUQDBPTJmEdks5krL9E-ZQepmGXt"},
    "friends_RC": {"dataset_class": friends_RC_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1Gi70GnNNRQWgnJNaOpbx9vVKq7L8Lbte"},
    "friends_QA": {"dataset_class": friends_QA_dataset,
                   "url": "https://drive.google.com/uc?export=download&id=1WlpmRNoYW5zXrOqBNw0OhVjyZ-eFXMCm"}
}
