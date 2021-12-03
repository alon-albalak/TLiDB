from .TLiDB_dataset import TLiDB_Dataset
from TLiDB.metrics.all_metrics import Accuracy, F1

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
    """
    _dataset_name = 'DailyDialog'
    _tasks = [
        'emotion_recognition', 'dialogue_act_classification', 'topic_classification',
        'causal_emotion_span_extraction', 'causal_emotion_entailment',
        'dialogue_nli', 'dialogue_reasoning_span_extraction', 'dialogue_reasoning_multiple_choice_span_selection',
        'dialogue_reasoning_commonsense_relation_prediction', 'adversarial_response_selection'
        ]
    _url = "https://drive.google.com/uc?export=download&id=1U9dUi16RbAprUiSBmEKnEpwk45USfnml"
    def __init__(self, task, dataset_folder, output_type, split=None):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self._dataset_name, task, output_type, dataset_folder=dataset_folder)
        
    def get_input(self, idx):
        pass

    def get_metadata(self, idx):
        pass

    def _collate_categorical(self, batch):
        pass

    def _collate_token(self, batch):
        pass

    def eval(self, y_pred, y_true, prediction_fn=None):
        pass