from .TLiDB_dataset import TLiDB_Dataset
from TLiDB.metrics.all_metrics import Accuracy, F1

class friends_ER_dataset(TLiDB_Dataset):
    """
    Friends ER dataset
    This is the full dataset from https://github.com/emorynlp/emotion-detection

    Input (x):
        - text (str): Text utterance

    Target (y):
        - label (str): Emotion label

    Metadata:
        - domain (str): Domain of the utterance

    """
    _dataset_name = "friends_ER"
    _tasks = ["emotion_recognition", "dialogue_response_generation"]
    _url = "https://drive.google.com/uc?export=download&id=1hjbtUUQDBPTJmEdks5krL9E-ZQepmGXt"
    def __init__(self, task, dataset_folder, output_type, split=None, full_dialogue=True):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self.dataset_name, task, output_type, dataset_folder=dataset_folder)

        def anonymize_speakers(speaker, speaker_dict):
            anon_token="speaker"
            if speaker in speaker_dict:
                return speaker_dict[speaker]
            else:
                speaker_dict[speaker] = f"{anon_token}{len(speaker_dict)}"
                return speaker_dict[speaker]

        # initialize task data and metadata
        self._input_array = []
        self._y_array = []
        self._metadata_fields = ["domains", "labels"]
        self._metadata_array = [[] for _ in self._metadata_fields]
        self._metadata_array[self._metadata_fields.index("labels")] = self.task_metadata[task]['labels']

        for datum in self.dataset['data']:
            if split and datum['dialogue_metadata']['original_data_partition'] != split:
                continue
            dialogue = ""
            speaker_dict = {}
            for turn in datum['dialogue']:
                utterance = ""
                for speaker in turn['speakers']:
                    # anonymize speakers
                    # dialogue += speaker
                    utterance += anonymize_speakers(speaker, speaker_dict)
                utterance += f": {turn['utterance']} "

                # add utterances to dialogue in reverse order, so most recent is first
                dialogue = utterance + dialogue

                # if using the full dialogue, add dialogue
                if full_dialogue:
                    self._input_array.append(dialogue)
                # else add only this utterance
                else:
                    self._input_array.append(utterance[:-1])
                
                self._y_array.append(turn['emotion_recognition'])
                self.get_metadata_field("domains").append("friends")
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
            y.append(item[1])
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
        accuracy_metric = Accuracy(prediction_fn=prediction_fn)
        f1_metric = F1(prediction_fn=prediction_fn)

        results = accuracy_metric.compute(y_pred, y_true)
        results.update(f1_metric.compute(y_pred, y_true))

        results_str = (
            f"Average {f1_metric.name}: {results[f1_metric.agg_metric_field]:0.4f}\n"
            f"Average {accuracy_metric.name}: {results[accuracy_metric.agg_metric_field]:0.4f}"
        )
        return results, results_str