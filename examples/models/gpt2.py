from torch.nn import CrossEntropyLoss
from .TLiDB_model import TLiDB_model

class GPT2(TLiDB_model):
    _encoder_only = False
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer, self.model = initialize_model(config)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(config.model, pad_token='<pad>')
        # self.model = GPT2LMHeadModel.from_pretrained(config.model)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.transformer = self.model.transformer
        self.lm_head = self.model.lm_head
        self.layers = [self.transformer, self.lm_head]
    
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
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            y_true = self.tokenizer.batch_decode(lm_labels, skip_special_tokens=True)
        pred_tokens = self.decode_logits(lm_logits)

        return pred_tokens, loss, y_true

    def transform_inputs(self, inputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs

    def transform_outputs(self, outputs, task,dataset_name):
        """tokenizes outputs"""
        tokenized_outputs = self.tokenizer(outputs, padding="max_length", truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_outputs.input_ids

    def decode_logits(self, logits):
        assert logits.dim() > 1
        pred_tokens = logits.argmax(-1)
        return self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

def initialize_model(config):
    if 'neo' in config.model:
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config.model, pad_token='<pad>')
        model = GPTNeoForCausalLM.from_pretrained(config.model)
        return tokenizer, model
    else:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config.model, pad_token='<pad>')
        model = GPT2LMHeadModel.from_pretrained(config.model)
        return tokenizer, model