from torch.nn import CrossEntropyLoss
from .TLiDB_model import TLiDB_model
import torch

# TODO: Follow patterns for
#   Transfertransfo: https://github.com/huggingface/transfer-learning-conv-ai/blob/master/train.py
#   Generate: https://huggingface.co/docs/transformers/master/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate
#             https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py#L649
#   GPT2 model: https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py

class GPT2(TLiDB_model):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer, self.model = initialize_model(config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.layers = {"model":self.model}
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

    def _forward(
        self,
        input_ids = None,
        attention_mask = None,
        lm_labels = None
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels)
        return outputs.loss


    def transform_LM_inputs(self, inputs, outputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer([" ".join([i,o])+self.tokenizer.eos_token for i,o in zip(inputs,outputs)], padding="longest",truncation=True, return_tensors="pt")
        labels = tokenized_inputs.input_ids.detach().clone()
        # replace pad tokens by -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return tokenized_inputs, labels

    def transform_generation_inputs(self, inputs):
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")
        self.tokenizer.padding_side = "right"
        return tokenized_inputs


    def generate(self, X, **kwargs):
        input_size = X.input_ids.shape[-1]
        outputs = self.model.generate(input_ids=X.input_ids,attention_mask=X.attention_mask,
                                    max_length=input_size+20, **kwargs)
        pred_tokens = outputs[:, input_size:]
        preds = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


def initialize_model(config):
    if 'neo' in config.model:
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config.model, pad_token='<|pad|>')
        model = GPTNeoForCausalLM.from_pretrained(config.model)
        return tokenizer, model
    else:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained(config.model, pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(config.model)
        model.config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model