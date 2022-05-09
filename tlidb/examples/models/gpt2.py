import random
from .TLiDB_model import TLiDB_model

class GPT2(TLiDB_model):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer, self.model = initialize_model(config)
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
        # if multiple correct outputs, select 1 at random
        if isinstance(outputs[0], list):
            outputs = [random.choice(output) for output in outputs]
        tokenized_inputs = self.tokenizer([" ".join([i,o])+self.tokenizer.eos_token for i,o in zip(inputs,outputs)],
                                        padding="longest",pad_to_multiple_of=8,truncation=True, return_tensors="pt")
        labels = tokenized_inputs.input_ids.detach().clone()
        # replace pad tokens by -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return tokenized_inputs, labels

    def transform_generation_inputs(self, inputs):
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(inputs, padding="longest", pad_to_multiple_of=8,
                                        truncation=True, return_tensors="pt")
        self.tokenizer.padding_side = "right"
        return tokenized_inputs


    def generate(self, X, max_decode_tokens, **kwargs):
        input_size = X.input_ids.shape[-1]
        outputs = self.model.generate(input_ids=X.input_ids,attention_mask=X.attention_mask,
                                    max_length=input_size+max_decode_tokens, **kwargs)
        pred_tokens = outputs[:, input_size:]
        preds = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


def initialize_model(config):
    if 'neo' in config.model:
        from transformers import GPTNeoForCausalLM as LLM
        from transformers import GPT2Tokenizer as Tokenizer
    else:
        from transformers import GPT2LMHeadModel as LLM
        from transformers import GPT2TokenizerFast as Tokenizer

    tokenizer = Tokenizer.from_pretrained(config.model, pad_token='<|pad|>')
    model = LLM.from_pretrained(config.model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
