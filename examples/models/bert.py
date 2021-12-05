from transformers import BertModel, BertTokenizerFast
import torch
from .TLiDB_model import TLiDB_model
from examples.utils import concat_t_d

SEQUENCE_TASKS = [
    'emotion_recognition', 'intent_detection', 'intent_classification',
    'dialogue_act_classification', 'topic_classification']
TOKEN_TASKS = []

class Bert(TLiDB_model):
    def __init__(self, config, datasets):
        super().__init__(config)
        self.tokenizer = get_bert_tokenizer(config.model)
        self.model = get_bert_model(config.model)
        self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)
        self.layers = {"model":self.model}
        # for each task/domain, we add a new classification layer to the model
        self.classifiers = {}
        for split in datasets.keys():
            for d in datasets[split]['datasets']:
                if d.task not in SEQUENCE_TASKS+TOKEN_TASKS:
                    raise ValueError('Task {} not supported by Bert'.format(d.task))
                t_d = concat_t_d(d.task,d.dataset_name)
                if t_d not in self.classifiers.keys():
                    setattr(self, f"{t_d}_classifier", torch.nn.Linear(self.model.config.hidden_size, d.num_classes))
                    forward = self.sequence_classification if d.task in SEQUENCE_TASKS else self.token_classification
                    self.classifiers[t_d] = {
                        "classifier": getattr(self, f"{t_d}_classifier"),
                        "labels":d.task_labels,
                        "forward":forward}
                    self.layers[f"{t_d}_classifier"] = getattr(self, f"{t_d}_classifier")

    def load_state_dict(self, state_dict):
        for layer_name, layer in state_dict.items():
            self.layers[layer_name].load_state_dict(layer)

    def _forward(self, inputs, task, dataset_name):
        return self.classifiers[concat_t_d(task,dataset_name)]['forward'](inputs, task, dataset_name)

    def transform_inputs(self, inputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs
    
    def transform_outputs(self, outputs, task,dataset_name):
        """Calls the classification layer associated with task and dataset_name"""
        t_d = concat_t_d(task,dataset_name)
        outputs = [self.classifiers[t_d]['labels'].index(y) for y in outputs]
        return torch.tensor(outputs, dtype=torch.long)

    # classify a sequence
    def sequence_classification(self, tokenized_sequences, task, dataset_name):
        """Classify a sequence of tokens with a single label"""
        t_d = concat_t_d(task,dataset_name)
        outputs = self.model(**tokenized_sequences)['pooler_output']
        outputs = self.dropout(outputs)
        logits = self.classifiers[t_d]['classifier'](outputs)
        return logits

    # classify a sequence of tokens
    def token_classification(self, tokenized_sequences, task, dataset_name):
        """Classify each token in a sequence"""
        t_d = concat_t_d(task,dataset_name)
        outputs = self.model(**tokenized_sequences)['last_hidden_states']
        outputs = self.dropout(outputs)
        logits = self.classifiers[t_d]['classifier'](outputs)
        return outputs

def get_bert_tokenizer(model):
    if model in ["bert-base-uncased"]:
        tokenizer = BertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f"Unsupported tokenizer model: {model}")
    return tokenizer

def get_bert_model(model):
    if model in ["bert-base-uncased"]:
        model = BertModel.from_pretrained(model)
    else:
        raise ValueError(f"Unsupported BERT model: {model}")
    return model

