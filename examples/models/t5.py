from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from .TLiDB_model import TLiDB_model

# # TLiDB_model specific imports
# from itertools import chain
# class TLiDB_model:
#     def __init__(self, config):
#         self.config = config

#     @property
#     def model(self):
#         """
#         Base model underlying the encoding/decoding
#         model is of type transformers.PreTrainedModel
#         """
#         return self._model
    
#     @model.setter
#     def model(self, model):
#         self._model = model

#     @property
#     def layers(self):
#         return self._layers

#     @layers.setter
#     def layers(self, layers):
#         self._layers = layers

#     @layers.getter
#     def layers(self):
#         return self._layers

#     def parameters(self):
#         """Convenience function to gather all model parameters
#             to be passed to a torch.optim optimizer"""
#         params = []
#         for layer in self.layers:
#             params.extend(layer.parameters())
#         return params

#     def named_parameters(self):
#         """Convenience function to gather all named model parameters
#             to be passed to a torch.optim optimizer"""
#         named_params = chain()
#         for layer in self.layers:
#             named_params = chain(named_params, layer.named_parameters())
#         return named_params

#     def zero_grad(self, set_to_none=True):
#         """Convenience function to zero gradients for all layers"""
#         for layer in self.layers:
#             layer.zero_grad(set_to_none=set_to_none)

#     @property
#     def forward(self):
#         return NotImplementedError

#     @forward.setter
#     def forward(self, forward):
#         self._forward = forward

#     def __call__(self, *args):
#         return self._forward(*args)

#     def transform_inputs(self, inputs):
#         return NotImplementedError

#     def transform_outputs(self, outputs):
#         return NotImplementedError

#     def to(self, device):
#         """Convenience function to move all layers to a device"""
#         for layer in self.layers:
#             layer.to(device)


# # BERT specific stuff
# from transformers import BertTokenizerFast, BertModel
# from examples.utils import concat_t_d

# SEQUENCE_TASKS = ['intent_detection', 'emotion_recognition']
# TOKEN_TASKS = []

# class Bert(TLiDB_model):
#     def __init__(self, config, datasets):
#         super().__init__(config)
#         self.tokenizer = get_bert_tokenizer(config.model)
#         self.model = get_bert_model(config.model)
#         self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)
#         self.layers = [self.model]
#         # for each task/domain, we add a new classification layer to the model
#         self.classifiers = {}
#         for split in datasets.keys():
#             for d in datasets[split]['datasets']:
#                 if d.task not in SEQUENCE_TASKS+TOKEN_TASKS:
#                     raise ValueError('Task {} not supported by Bert'.format(d.task))
#                 t_d = concat_t_d(d.task,d.dataset_name)
#                 if t_d not in self.classifiers.keys():
#                     setattr(self, f"{t_d}_classifier", torch.nn.Linear(self.model.config.hidden_size, d.num_classes))
#                     forward = self.sequence_classification if d.task in SEQUENCE_TASKS else self.token_classification
#                     self.classifiers[t_d] = {
#                         "classifier": getattr(self, f"{t_d}_classifier"),
#                         "labels":d.get_metadata_field("labels"),
#                         "forward":forward}
#                     self.layers.append(getattr(self, f"{t_d}_classifier"))

#     def _forward(self, inputs, task, dataset_name):
#         return self.classifiers[concat_t_d(task,dataset_name)]['forward'](inputs, task, dataset_name)

#     def transform_inputs(self, inputs):
#         """Only tokenizes inputs"""
#         tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
#         return tokenized_inputs
    
#     def transform_outputs(self, outputs, task,dataset_name):
#         """Calls the classification layer associated with task and dataset_name"""
#         t_d = concat_t_d(task,dataset_name)
#         outputs = [self.classifiers[t_d]['labels'].index(y) for y in outputs]
#         return torch.tensor(outputs, dtype=torch.long)

#     # classify a sequence
#     def sequence_classification(self, tokenized_sequences, task, dataset_name):
#         """Classify a sequence of tokens with a single label"""
#         t_d = concat_t_d(task,dataset_name)
#         outputs = self.model(**tokenized_sequences)['pooler_output']
#         outputs = self.dropout(outputs)
#         logits = self.classifiers[t_d]['classifier'](outputs)
#         return logits

#     # classify a sequence of tokens
#     def token_classification(self, tokenized_sequences, task, dataset_name):
#         """Classify each token in a sequence"""
#         t_d = concat_t_d(task,dataset_name)
#         outputs = self.model(**tokenized_sequences)['last_hidden_states']
#         outputs = self.dropout(outputs)
#         logits = self.classifiers[t_d]['classifier'](outputs)
#         return outputs

# def get_bert_tokenizer(model):
#     if model in ["bert-base-uncased"]:
#         tokenizer = BertTokenizerFast.from_pretrained(model)
#     else:
#         raise ValueError(f"Unsupported tokenizer model: {model}")
#     return tokenizer

# def get_bert_model(model):
#     if model in ["bert-base-uncased"]:
#         model = BertModel.from_pretrained(model)
#     else:
#         raise ValueError(f"Unsupported BERT model: {model}")
#     return model



class T5(TLiDB_model):
    _requires_y_true = True
    def __init__(self, config, datasets):
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