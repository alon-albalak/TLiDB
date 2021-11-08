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
    _output_processing_function="multiclass_logits_to_pred"
    def __init__(self, task, dataset_folder, output_type, split=None):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self.dataset_name, task, output_type, dataset_folder=dataset_folder)
        
        # initialize task data and metadata
        self._input_array = []
        self._y_array = []
        self._metadata_fields = ["domains", "labels"]
        self._metadata_array = [[] for _ in self._metadata_fields]
        self._metadata_array[self._metadata_fields.index("labels")] = self.task_metadata[task]['labels']

        for datum in self.dataset['data']:
            if split and datum['dialogue_metadata']['original_data_partition'] != split:
                continue
            utterance = datum['dialogue'][0]
            domain = utterance['intent_detection']['domain']
            intent = utterance['intent_detection']['intent']
            self._input_array.append(utterance['utterance'])
            self._y_array.append([domain, intent])
            self.get_metadata_field("domains").append(domain)

        self._num_classes = len(self.get_metadata_field("labels"))
        self._y_size = len(self._y_array)

    def get_input(self, idx):
        return self._input_array[idx]

    def get_metadata(self, idx):
        return {"domains": self.get_metadata_field("domains")[idx]}

    def _collate_categorical(self, batch):
        X,y, metadata = [], [], {}
        for item in batch:
            X.append(item[0])
            y.append(f"{item[1][0]}_{item[1][1]}")
            for k, v in item[2].items():
                if k not in metadata:
                    metadata[k] = []
                metadata[k].append(v)
        return X,y, metadata

    def _collate_token(self, batch):
        X,y, metadata = [], [], {}
        for item in batch:
            X.append(item[0])
            y.append(item[1])
            for k, v in item[2].items():
                if k not in metadata:
                    metadata[k] = []
                metadata[k].append(v)
        return X,y, metadata

    def eval(self, y_pred, y_true, prediction_fn=None):
        """
        Evaluates the model performance
        Args:
            - y_pred (Tensor): Predicted targets
            - y_true (LongTensor): True targets
            - prediction_fn (function): Function to convert y_pred into predicted labels
        Returns:
            - results (dict): Dictionary of evaluation metrics
            - results_str (str): Summary string of evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(metric, y_pred, y_true)