import os
import json
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import numpy as np
from torch.utils.data import Dataset


def download_and_unzip(url, extract_to='.'):
    print(f"Waiting for response from {url}")
    http_response = urlopen(url)
    print("Downloading data from {url}")
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

def load_dataset(name, dataset_folder, url):
    # download and unzip dataset if needed
    if f"TLiDB_{name}" not in os.listdir(dataset_folder):
        assert(url is not None), "Must provide a url to download from"
        download_and_unzip(url, dataset_folder)
        print(f"Extracted files to {dataset_folder}/{name}")

    ds = load_dataset_local(name, dataset_folder)

    return ds

class TLiDB_Dataset(Dataset):
    """
    Abstract dataset class for all TLiDB datasets
    """
    def __init__(self, dataset_name, task, output_type, dataset_folder):
        super().__init__()
        self.dataset = load_dataset(dataset_name, dataset_folder, self.url)
        self._task = task
        self.task_metadata = self.dataset['metadata']['task_metadata']

        if task == "response_generation":
            #TODO:
            # gather data such that _input_array contains all utterances up to the current one
            # and _y_array contains the response
            pass
        else:
            self.metrics = self.task_metadata[task]['metrics']

        if output_type == "categorical":
            self._collate = self._collate_categorical
        elif output_type == "token":
            self._collate = self._collate_token
        else:
            raise ValueError(f"{output_type} is not a valid output type")

    @property
    def dataset_name(self):
        return self._dataset_name
    
    @property
    def tasks(self):
        return self._tasks

    @property
    def task(self):
        return self._task

    @property
    def url(self):
        return self._url

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

    @property
    def metadata_fields(self):
        """
        Returns the fields that are stored in the metadata
        Metadata should always contain the domains
        If it is a classification task, then metadata should also contain the classes
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        Returns the metadata array
        """
        return self._metadata_array

    @property
    def output_processing_function(self):
        """
        Returns the function to convert model outputs into predictions
        """
        return getattr(self, "_output_processing_function", None)

    def get_metadata_field(self, field):
        return self.metadata_array[self.metadata_fields.index(field)]

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset
        """
        x = self.get_input(idx)
        y = self.get_target(idx)
        m = self.get_metadata(idx)
        return x, y, m

    def get_input(self, idx):
        return NotImplementedError

    def get_target(self, idx):
        return self.y_array[idx]

    def get_metadata(self, idx):
        return NotImplementedError

    def _collate(self, batch):
        return NotImplementedError

    def _collate_categorical(self, batch):
        return NotImplementedError
    
    def _collate_token(self, batch):
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

    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:0.4f}\n"
        )
        return results, results_str

    def random_subsample(self, frac=1.0):
        """
        Subsamples the dataset
        Args:
            - frac (float): Fraction of the dataset to keep
        """
        if frac < 1.0:
            num_to_retain = int(self.y_size * frac)
            idxs_to_retain = np.sort(np.random.permutation(len(self))[:num_to_retain]).tolist()
            subsampled_input_array, subsampled_y_array, subsampled_metadata_array = [], [], []
            for idx in idxs_to_retain:
                input_item, y_item, metadata_item = self.__getitem__(idx)
                subsampled_input_array.append(input_item)
                subsampled_y_array.append(y_item)
                subsampled_metadata_array.append(metadata_item)
            self._input_array = subsampled_input_array
            self._y_array = subsampled_y_array
            metadata_iterated = list(metadata_item.keys())
            metadata_not_iterated = [metadata_field for metadata_field in self.metadata_fields if metadata_field not in metadata_iterated]
            subsampled_metadata_array = [subsampled_metadata_array]
            for metadata_field in metadata_not_iterated:
                subsampled_metadata_array.append(self.get_metadata_field(metadata_field))
            self._metadata_array = subsampled_metadata_array
            self._metadata_fields = metadata_iterated+metadata_not_iterated
            self._y_size = num_to_retain

class friends_ER_dataset(TLiDB_Dataset):
    _dataset_name = "friends_ER"

class friends_RC_dataset(TLiDB_Dataset):
    _dataset_name = "friends_RC"

class friends_QA_dataset(TLiDB_Dataset):
    _dataset_name = "friends_QA"

class banking77_datset(TLiDB_Dataset):
    _dataset_name = "banking77"

class multidogo_datset(TLiDB_Dataset):
    _dataset_name = "multidogo"