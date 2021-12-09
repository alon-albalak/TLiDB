from transformers import BertModel, BertTokenizerFast
import torch
from .TLiDB_model import TLiDB_model
from examples.utils import concat_t_d

SEQUENCE_TASKS = [
    'emotion_recognition', 'intent_detection', 'intent_classification',
    'dialogue_act_classification', 'topic_classification', 'causal_emotion_entailment',
    'dialogue_nli', "dialogue_reasoning_commonsense_relation_prediction"]
TOKEN_TASKS = []
SPAN_EXTRACTION_TASKS = [
    "causal_emotion_span_extraction", "dialogue_reasoning_span_extraction"
]
MULTIPLE_CHOICE_TASKS = ["dialogue_reasoning_multiple_choice_span_selection",
    "adversarial_response_selection"]

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
                # make this more flexible, call a layer initializer which knows what task to create the layer for
                t_d = concat_t_d(d.task,d.dataset_name)
                if t_d not in self.classifiers.keys():
                    layer = self.initialize_bert_classifier(d, config)

                    setattr(self, f"{t_d}_classifier", layer)
                    forward = self.initialize_forward(d.task)
                    self.classifiers[t_d] = {
                        "classifier": getattr(self, f"{t_d}_classifier"),
                        "labels":d.task_labels,
                        "forward":forward}
                    self.layers[f"{t_d}_classifier"] = getattr(self, f"{t_d}_classifier")

        self.init_weights()

    def initialize_bert_classifier(self, dataset, config):
        if dataset.task in SEQUENCE_TASKS:
            return torch.nn.Linear(self.model.config.hidden_size, dataset.num_classes)
        elif dataset.task in TOKEN_TASKS:
            return torch.nn.Linear(self.model.config.hidden_size, dataset.num_classes)
        elif dataset.task in SPAN_EXTRACTION_TASKS:
            return torch.nn.Linear(self.model.config.hidden_size, 2)
        elif dataset.task in MULTIPLE_CHOICE_TASKS:
            return torch.nn.Linear(self.model.config.hidden_size, 1)
        else:
            raise ValueError(f"Unsupported task: {dataset.task}")

    def initialize_forward(self, task):
        if task in SEQUENCE_TASKS:
            return self.sequence_classification
        elif task in TOKEN_TASKS:
            return self.token_classification
        elif task in SPAN_EXTRACTION_TASKS:
            return self.span_extraction
        elif task in MULTIPLE_CHOICE_TASKS:
            return self.multiple_choice
        else:
            raise ValueError(f"Unsupported task: {task}")

    def load_state_dict(self, state_dict):
        for layer_name, layer in state_dict.items():
            if layer_name in self.layers.keys():
                self.layers[layer_name].load_state_dict(layer)

    def _forward(self, inputs, task, dataset_name):
        return self.classifiers[concat_t_d(task,dataset_name)]['forward'](inputs, task, dataset_name)

    def transform_inputs(self, inputs, metadata):
        """Only tokenizes inputs"""
        # specify if we need the offset mapping
        if 'return_offsets_mapping' in metadata:
            return_offsets_mapping = metadata['return_offsets_mapping']
        else:
            return_offsets_mapping = False

        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length,\
                return_offsets_mapping=return_offsets_mapping,return_tensors="pt")
        return tokenized_inputs
    
    def transform_outputs(self, inputs, outputs, task_type, metadata):
        """Calls the classification layer associated with task and dataset_name"""
        t_d = concat_t_d(metadata['task'],metadata['dataset_name'])
        outputs = getattr(self, f"transform_{task_type}_outputs")(inputs, outputs, t_d, metadata)
        return outputs

    def transform_classification_outputs(self,inputs, outputs, t_d, metadata):
        outputs = [self.classifiers[t_d]['labels'].index(y) for y in outputs]
        return torch.tensor(outputs, dtype=torch.long)

    def transform_span_extraction_outputs(self, inputs, outputs, t_d, metadata):
        start_indices, end_indices = [], []

        for offset_mapping,output in zip(inputs.offset_mapping,outputs):
            start_idx,end_idx = get_token_offsets(offset_mapping, output['text'], output['answer_start'])
            start_indices.append(start_idx)
            end_indices.append(end_idx)
        
        return [torch.tensor(start_indices,dtype=torch.long), torch.tensor(end_indices,dtype=torch.long)]
    
    def transform_multiple_choice_outputs(self, inputs, outputs, t_d, metadata):
        return torch.tensor(outputs, dtype=torch.long)

    # classify a sequence
    def sequence_classification(self, tokenized_sequences, task, dataset_name):
        """Classify a sequence of tokens with a single label"""
        t_d = concat_t_d(task,dataset_name)
        outputs = self.model(input_ids=tokenized_sequences.input_ids, attention_mask=tokenized_sequences.attention_mask)['pooler_output']
        outputs = self.dropout(outputs)
        logits = self.classifiers[t_d]['classifier'](outputs)
        return logits

    # classify a sequence of tokens
    def token_classification(self, tokenized_sequences, task, dataset_name):
        """Classify each token in a sequence"""
        t_d = concat_t_d(task,dataset_name)
        outputs = self.model(input_ids=tokenized_sequences.input_ids, attention_mask=tokenized_sequences.attention_mask)['last_hidden_state']
        outputs = self.dropout(outputs)
        logits = self.classifiers[t_d]['classifier'](outputs)
        return logits

    # extract spans
    def span_extraction(self, tokenized_sequences, task, dataset_name):
        """Extract spans from a sequence of tokens"""
        t_d = concat_t_d(task,dataset_name)
        outputs = self.model(input_ids=tokenized_sequences.input_ids, attention_mask=tokenized_sequences.attention_mask)['last_hidden_state']
        outputs = self.dropout(outputs)
        logits = self.classifiers[t_d]['classifier'](outputs)
        return logits

    def multiple_choice(self, tokenized_sequences, task, dataset_name):
        """Multiple choice classification"""
        t_d = concat_t_d(task,dataset_name)
        outputs = self.model(input_ids=tokenized_sequences.input_ids, attention_mask=tokenized_sequences.attention_mask)['pooler_output']
        outputs = self.dropout(outputs)
        logits = self.classifiers[t_d]['classifier'](outputs)
        return logits

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

def get_token_offsets(offsets_mapping, text, text_start_idx):
    """
    Given a start index and length of the text, return the start and end indices of the span
    """
    if text_start_idx < 0:
        return 0,0
    token_start_idx, token_end_idx = 0,0
    text_end_index = text_start_idx + len(text)
    for i, (start, end) in enumerate(offsets_mapping):
        if not token_start_idx and start <= text_start_idx and end >= text_start_idx:
            token_start_idx = i
        if not token_end_idx and start <= text_end_index and end >= text_end_index:
            token_end_idx = i
    return token_start_idx, token_end_idx