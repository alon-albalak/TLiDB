from .TLiDB_dataset import TLiDB_Dataset
from TLiDB.metrics.all_metrics import Accuracy, F1

# TODO:
# - add support for multiple choice - https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMultipleChoice

class DailyDialog_dataset(TLiDB_Dataset):
    """
    DailyDialog dataset
    This dataset contains all available annotations (currently) from:
        - DailyDialog - http://yanran.li/files/ijcnlp_dailydialog.zip
            - emotion recognition
            - dialog act classification
            - topic classification
        - RECCON - https://github.com/declare-lab/RECCON
            - causal emotion span extraction
            - causal entailment of emotion
        - CIDER - https://github.com/declare-lab/CIDER
            - dialogue level NLI
            - Dialogue reasoning span extraction
            - Dialogue reasoning multiple choice span selection
            - Commonsense relation prediction
        - DailyDialog++ - https://iitmnlp.github.io/DailyDialog-plusplus/
            - Adversarial response selection

    Metadata:
        13118 Total dialogues
        Dialogues per task:
            Emotion recognition: 13118 Dialogues
            Dialog act classification: 13118 Dialogues
            Topic classification: 13118 Dialogues
            Causal emotion span extraction: 1106 Dialogues
            Causal entailment of emotion: 1106 Dialogues
            Dialogue level NLI: 245 Dialogues
            Dialogue reasoning span extraction: 227 Dialogues
            Dialogue reasoning multiple choice span selection: 226 Dialogues
            Commonsense relation prediction: 245 Dialogues
            Adversarial response selection: 6880 Dialogues
    """
    _dataset_name = 'DailyDialog'
    _tasks = [
        'emotion_recognition', 'dialogue_act_classification', 'topic_classification',
        'causal_emotion_span_extraction', 'causal_emotion_entailment',
        'dialogue_nli', 'dialogue_reasoning_span_extraction', 'dialogue_reasoning_multiple_choice_span_selection',
        'dialogue_reasoning_commonsense_relation_prediction', 'adversarial_response_selection'
        ]
    _url = "https://drive.google.com/uc?export=download&id=1U9dUi16RbAprUiSBmEKnEpwk45USfnml"
    _task_metadatas = {
        "emotion_recognition": {
                "prompt":"emotion:","type":"classification","annotation_type":"utterance_level_classification"
                },
        "dialogue_act_classification": {
            "prompt":"dialogue act:","type":"classification","annotation_type":"utterance_level_classification"
            },
        "topic_classification":{
            "prompt":"topic:","type":"classification","annotation_type":"dialogue_level_classification"
            },
        "causal_emotion_span_extraction":{
            "prompt":"","type":"span_extraction","annotation_type":"span_extraction",
            },
    }
    def __init__(self, task, dataset_folder, model_type, split=None):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self._dataset_name, task, model_type, dataset_folder=dataset_folder)
        self._task_metadata = self._task_metadatas[task]
        self._input_array = []
        self._y_array = []
        self._metadata_fields = []
        self._metadata_array = []
        self._load_data(task, split)
        self._num_classes = len(self.task_labels)
        self._y_size = len(self._y_array)
        

    def _load_data(self, task, split):
        # get the data loader, based on whether the task is utterance level or dialogue level
        loader = getattr(self, f"_load_{self._task_metadata['annotation_type']}_task")
        return loader(task,split)

    def _load_utterance_level_classification_task(self, task, split):
        for datum in self.dataset['data']:
            # TODO: create our own splits by task
            if split and datum['dialogue_metadata']['original_data_partition'] != split:
                continue
            dialogue = []
            for turn in datum['dialogue']:
                dialogue.append([turn['speakers'][0], turn['utterance']])
                if task in turn:
                    self._input_array.append(dialogue.copy())
                    self._y_array.append(turn[task])

    def _load_dialogue_level_classification_task(self, task, split):
        for datum in self.dataset['data']:
            # TODO: create our own splits by task
            if split and datum['dialogue_metadata']['original_data_partition'] != split:
                continue
            if task in datum:
                dialogue = [[turn['speakers'][0], turn['utterance']] for turn in datum['dialogue']]
                self._input_array.append(dialogue)
                self._y_array.append(datum[task])
        
    def _load_span_extraction_task(self, task, split):
        for datum in self.dataset['data']:
            # TODO: create our own splits by task
            if split and datum['dialogue_metadata']['original_data_partition'] != split:
                continue
            if task in datum:
                for qas in datum[task]:
                    for qa in qas['qas']:
                        for answer in qa['answers']:
                            self._input_array.append({
                                "context":qas['context'],"question":qa['question']
                            })
                            self._y_array.append(answer)

    def get_input(self, idx):
        return self._input_array[idx]

    def get_metadata(self, idx):
        return {}

    def _collate_encoder(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['type'] == 'span_extraction':
                X.append(self._join_strings(item[0]['context'],item[0]['question']))
            else:
                X.append(self._convert_dialogue_to_string(item[0]))
            y.append(item[1])
            for k, v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        return X, y, metadata

    def _collate_decoder(self, batch):
        labels = self.task_labels
        pass

    def _collate_seq2seq(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['type'] == "span_extraction":
                X.append(self._join_strings("context:",item[0]['context'],item[0]['question']))
            else:
                dialogue = self._convert_dialogue_to_string(item[0])
                X.append(self._join_strings("context:",dialogue,self._task_metadata['prompt']))                
            y.append(item[1])
            for k, v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        labels = self.task_labels
        if labels:
            metadata['labels'] = labels
        return X, y, metadata

    def _convert_dialogue_to_string(self, input):
        dialogue = ""
        for (speaker, utt) in input:
            dialogue += f"{speaker}: {utt} "
        return dialogue[:-1]

    def _join_strings(self, *args):
        return " ".join(args)