from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from .TLiDB_model import TLiDB_model

class T5(TLiDB_model):
    _requires_y_true = True
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = T5Tokenizer.from_pretrained(config.model)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.lm_head = self.model.lm_head
        self.layers = [self.encoder, self.decoder, self.lm_head]                
    
    def _forward(self, inputs, task, dataset_name, y_true=None):
        encoder_outputs = self.encoder(**inputs).last_hidden_state
        if y_true is not None:
            decoder_input_ids = self.model._shift_right(y_true)
        sequence_output = self.decoder(input_ids=decoder_input_ids,
                                        encoder_hidden_states=encoder_outputs,
                                        encoder_attention_mask=inputs['attention_mask'])[0]
        lm_logits = self.lm_head(sequence_output)

        return lm_logits

    def transform_inputs(self, inputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs

    def transform_outputs(self, outputs, task,dataset_name):
        """tokenizes outputs"""
        tokenized_outputs = self.tokenizer(outputs, padding="longest", truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_outputs.input_ids