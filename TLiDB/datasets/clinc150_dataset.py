from .TLiDB_dataset import TLiDB_Dataset
from TLiDB.metrics.all_metrics import Accuracy

class clinc150_dataset(TLiDB_Dataset):
    """
    CLINC150 dataset
    This is the full dataset from https://github.com/clinc/oos-eval

    Input (x):
        - text (str): Text utterance

    Target (y):
        - label (list): List of [Domain, Intent] labels

    Metadata:
        - domain (str): Domain of the utterance

    """
    _dataset_name = "clinc150"
    _tasks = ["intent_detection"]
    _url = "https://drive.google.com/uc?export=download&id=1dG6KXQ6L7xpbnWmhW9Xo3vPSfYstk43E"
    def __init__(self, task, dataset_folder, model_type, split=None):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self.dataset_name, task, model_type, dataset_folder=dataset_folder)
        
        # initialize task data and metadata
        categories = [
            "auto and commute","banking","credit cards","home",
            "kitchen and dining","meta","small talk","travel",
            "utility","work"
            ]
        self._input_array = []
        self._y_array = []
        self._metadata_fields = ["domains"]
        self._metadata_array = [[] for _ in self._metadata_fields]
        
        # convert labels to human readable
        labels = [label.replace("_"," ") for label in self.task_labels]
        formatted_labels = []
        for label in labels:
            for c in categories:
                if c == label[:len(c)]:
                    formatted_label = c+":"+label[len(c):]
                    formatted_labels.append(formatted_label)
        self.task_labels = formatted_labels

        for datum in self.dataset['data']:
            if split and datum['dialogue_metadata']['original_data_partition'] != split:
                continue
            utterance = datum['dialogue'][0]
            domain = utterance['intent_detection']['domain']
            intent = utterance['intent_detection']['intent']
            self._input_array.append(utterance['utterance'])
            self._y_array.append([domain, intent])
            self.get_metadata_field("domains").append(domain)

        self._num_classes = len(self.task_labels)
        self._y_size = len(self._y_array)

    def get_input(self, idx):
        return self._input_array[idx]

    def get_metadata(self, idx):
        return {
            "domains": self.get_metadata_field("domains")[idx],
            }

    def _collate_encoder(self, batch):
        X,y, metadata = [], [], {}
        for item in batch:
            X.append(item[0])
            y.append(f"{item[1][0].replace('_',' ')}: {item[1][1].replace('_',' ')}")
            for k, v in item[2].items():
                if k not in metadata:
                    metadata[k] = []
                metadata[k].append(v)
        return X,y, metadata

    def _collate_decoder(self, batch):
        X,y, metadata = [], [], {}
        for item in batch:
            X.append(item[0])
            y.append(f"{item[1][0].replace('_',' ')}: {item[1][1].replace('_',' ')}")
            for k, v in item[2].items():
                if k not in metadata:
                    metadata[k] = []
                metadata[k].append(v)
        labels = self.task_labels
        if labels:
            metadata['labels'] = labels
        return X,y, metadata

    def _collate_seq2seq(self, batch):
        X,y, metadata = [], [], {}
        for item in batch:
            X.append(item[0])
            y.append(f"{item[1][0].replace('_',' ')}: {item[1][1].replace('_',' ')}")
            for k, v in item[2].items():
                if k not in metadata:
                    metadata[k] = []
                metadata[k].append(v)
        labels = self.task_labels
        if labels:
            metadata['labels'] = labels
        return X,y, metadata