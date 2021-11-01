from transformers import BertModel, BertTokenizerFast
import torch
from examples.models.TLiDB_model import TLiDB_model

class Bert(TLiDB_model):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.tokenizer = get_bert_tokenizer(config.model)
        self.model = get_bert_model(config.model)
        self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, dataset.num_classes)
        self.layers = (self.model, self.classifier)
        self.labels = dataset.get_metadata_field('labels')

        if config.task in ['intent_detection']:
            self._forward = self.sequence_classification
        else:
            self._forward = self.token_classification

    def transform_inputs(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs
    
    def transform_outputs(self, outputs):
        outputs = [self.labels.index(y) for y in outputs]
        return torch.tensor(outputs, dtype=torch.long)

    # classify a sequence
    def sequence_classification(self, tokenized_sequences):
        outputs = self.model(**tokenized_sequences)['pooler_output']
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits

    # classify a sequence of tokens
    def token_classification(self, tokenized_sequences):
        outputs = self.model(**tokenized_sequences)['last_hidden_states']
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
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

