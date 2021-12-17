from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn import CrossEntropyLoss
from .TLiDB_model import TLiDB_model
import torch

class T5(TLiDB_model):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = T5Tokenizer.from_pretrained(config.model)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model)
        self.layers = {"model":self.model}
        self.init_weights()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

    def _forward(
        self,
        input_ids = None,
        attention_mask = None,
        encoder_outputs = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        lm_labels = None,
    ):

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             encoder_outputs=encoder_outputs,
                             labels=lm_labels)
        return outputs.loss

    def transform_inputs(self, inputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer(inputs, padding="longest",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs

    def transform_outputs(self, outputs):
        """tokenizes outputs"""
        tokenized_outputs = self.tokenizer(outputs, padding="longest", truncation=True, return_tensors="pt")
        # replace pad tokens by -100
        label_ids = tokenized_outputs.input_ids
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return label_ids

    # Not used
    # def greedy_decode_logits(self, logits):
    #     assert logits.dim() > 1
    #     pred_tokens = logits.argmax(-1)
    #     return pred_tokens

    def generate(self, input_ids, **kwargs):
        pred_tokens = self.model.generate(input_ids=input_ids, **kwargs)
        return pred_tokens

    def batch_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)