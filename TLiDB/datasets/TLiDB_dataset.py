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
    print(f"Downloading data from {url}")
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
        print(f"Extracted files to {os.path.join(dataset_folder,name)}")

    ds = load_dataset_local(name, dataset_folder)

    return ds

def load_split_ids(name, dataset_folder, split, few_shot_percent=None):
    if f"TLiDB_{name}" not in os.listdir(dataset_folder):
        raise ValueError("Dataset not found")
    if few_shot_percent and split!="test":
        ids_file = f"{dataset_folder}/TLiDB_{name}/TTiDB_{few_shot_percent}_percent_few_shot_{split}_ids.txt"
    else:
        ids_file = f"{dataset_folder}/TLiDB_{name}/TTiDB_{split}_ids.txt"
    with open(ids_file) as f:
        ids = f.read().splitlines()
    return ids

class TLiDB_Dataset(Dataset):
    """
    Abstract dataset class for all TLiDB datasets
    """
    def __init__(self, dataset_name, task, model_type, max_dialogue_length, dataset_folder):
        super().__init__()
        self.dataset = load_dataset(dataset_name, dataset_folder, self.url)
        self._task = task
        task_metadata = self.dataset['metadata']['task_metadata']
        self.task_labels = []
        self._max_dialogue_length = max_dialogue_length
        self._model_type = model_type
        if task in task_metadata and 'labels' in task_metadata[task]:
            self.task_labels = task_metadata[task]['labels']

        if task == "response_generation":
            self.metrics = ['token_f1', 'bleu', 'bert_score', 'distinct_ngrams']
            self.metric_kwargs = {
                "bleu": [{"ngram_order": 1}, {"ngram_order": 2}, {"ngram_order": 3}, {"ngram_order": 4}],
                "distinct_ngrams": [{"ngram_order": 1}, {"ngram_order": 2}, {"ngram_order": 3}]
            }
            self._collate = self._collate_response_generation
        else:
            self.metrics = task_metadata[task]['metrics']
            self.metric_kwargs = task_metadata[task].get("metric_kwargs", dict())

            if model_type == "Encoder":
                self._collate = self._collate_encoder
            elif model_type == "Decoder":
                self._collate = self._collate_decoder
            elif model_type == "EncoderDecoder":
                self._collate = self._collate_encoderdecoder
            else:
                raise ValueError(f"{model_type} is not a valid algorithm type")

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
    def task_metadata(self):
        return self._task_metadata

    @property
    def url(self):
        return self._url

    @property
    def max_dialogue_length(self):
        return self._max_dialogue_length

    @property
    def model_type(self):
        return self._model_type

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
        return self._input_array[idx]

    def get_target(self, idx):
        return self.y_array[idx]

    def get_metadata(self, idx):
        return {}

    def _collate(self, batch):
        return NotImplementedError

    def _collate_encoder(self, batch):
        return NotImplementedError
    
    def _collate_decoder(self, batch):
        return NotImplementedError

    def _collate_encoderdecoder(self, batch):
        return NotImplementedError

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.y_array)

    def _truncate_dialogue(self, input):
        """
        Truncates the dialogue to the max dialogue length
        """
        if self.max_dialogue_length:
            dialogue = self._convert_dialogue_to_string(input)
            while len(dialogue.split()) > self.max_dialogue_length:
                input = input[1:]
                dialogue = self._convert_dialogue_to_string(input)

        return input

    def _convert_dialogue_to_string(self, input):
        dialogue = ""
        for (speaker, utt) in input:
            if speaker:
                dialogue += f"{speaker}: "
            dialogue += f"{utt} "
        return dialogue[:-1]

    def _join_strings(self, *args):
        return " ".join(args)

    def _load_response_generation_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                dialogue = []
                for turn in datum['dialogue']:
                    truncated_dialogue = self._truncate_dialogue(dialogue)
                    if turn['speakers']:
                        str_dialogue = self._convert_dialogue_to_string(truncated_dialogue)
                        str_dialogue += f" {' '.join(turn['speakers'])}: "
                        str_dialogue = str_dialogue.lstrip()
                        self._input_array.append(str_dialogue)
                        self._y_array.append(turn['utterance'])
                    dialogue.append([" ".join(turn['speakers']), turn['utterance']])

    def _collate_response_generation(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            X.append(item[0])
            y.append(item[1])
            for k, v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        return X, y, metadata

    def random_subsample(self, frac=1.0):
        """
        Subsamples the dataset
        Args:
            - frac (float): Fraction of the dataset to keep
        """
        if frac < 1.0:
            num_to_retain = int(self.y_size * frac)
            if num_to_retain == 0:
                return
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
