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
        self.transformer = self.model.transformer
        self.lm_head = self.model.lm_head
        self.layers = {"model":self.model}
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.transformer = self.model.transformer
        self.lm_head = self.model.lm_head

    def _forward(
        self,
        input_ids = None,
        attention_mask = None,
        lm_labels = None
    ):

        transformer_outputs = self.transformer(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        lm_logits = self.lm_head(transformer_outputs)
        loss = None
        y_true = None
        if lm_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            y_true = self.tokenizer.batch_decode(lm_labels, skip_special_tokens=True)

        # pred_tokens = self.decode_logits(lm_logits)

        return lm_logits, loss, y_true

    def transform_inputs(self, inputs, outputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer([i+o for i,o in zip(inputs,outputs)], padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs

    def transform_outputs(self, inputs, outputs):
        """tokenizes outputs"""
        tokenized_inputs = self.tokenizer(inputs)
        tokenized_outputs = self.tokenizer(outputs)
        outputs = [t+o for t,o in zip(tokenized_inputs,tokenized_outputs)]
        for output in outputs:
            while len(output) < self.config.max_seq_length:
                output.append(self.tokenizer.pad_token_id)

        # tokenized_outputs = self.tokenizer(outputs, padding="max_length", truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_outputs

    def preprocess_batch_for_forward(self, inputs, outputs):
        processed_inputs = self.tokenizer([i+o for i,o in zip(inputs,outputs)], padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")

        tokenized_inputs = self.tokenizer(inputs).input_ids
        tokenized_outputs = self.tokenizer(outputs).input_ids
        processed_outputs = [t+o for t,o in zip(tokenized_inputs,tokenized_outputs)]
        for i, output in enumerate(processed_outputs):
            while len(output) < self.config.max_seq_length:
                output.append(self.tokenizer.pad_token_id)
            if len(output) > self.config.max_seq_length:
                processed_outputs[i] = output[:self.config.max_seq_length]

        return processed_inputs, torch.tensor(processed_outputs)

    def preprocess_batch_for_generation(self, inputs):
        return self.tokenizer(inputs, padding=True, return_tensors='pt').input_ids

    def predict_class(self, input_ids, **kwargs):
        outputs = []
        for input_id in input_ids:
            output = self.model.generate(input_id.unsqueeze(0), pad_token_id=self.tokenizer.pad_token_id, max_length=input_ids.size(-1)+20, **kwargs)
            outputs.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        return outputs

    def decode_logits(self, logits):
        assert logits.dim() > 1
        pred_tokens = logits.argmax(-1)
        return self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

def initialize_model(config):
    if 'neo' in config.model:
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config.model, pad_token='<|pad|>')
        model = GPTNeoForCausalLM.from_pretrained(config.model)
        return tokenizer, model
    else:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config.model, pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(config.model)
        return tokenizer, model