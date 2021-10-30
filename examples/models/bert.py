from transformers import BertModel, BertTokenizerFast
import torch

from examples.models.TLiDB_model import TLiDB_model

# LEFT OFF: need to figure out how to handle multiple types of "__call__"
# some models will be sequence classification, some will be sequence tagging, etc.


class Bert(TLiDB_model):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = get_bert_tokenizer(config.model)
        self.model = get_bert_model(config.model)
        if config.task in ['intent_detection']:
            # TODO: add classification nn layer
            self.output_layer = self.sequence_classification

    def __call__(self, x):
        # TODO: make this a call to a function which can be used with 
        outputs = self.model(x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        outputs = self.output_layer(outputs)

        return outputs

    def transform_inputs(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs
    
    def transform_outputs(self, outputs):
        return torch.tensor(outputs, dtype=torch.long)

    # classify a sequence
    def sequence_classification(self, outputs):
        # TODO: figure this out
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

