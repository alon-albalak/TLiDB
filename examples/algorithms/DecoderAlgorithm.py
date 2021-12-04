from utils import move_to
from .algorithm import Algorithm
import torch

# TODO: Follow patterns for
#   Transfertransfo: https://github.com/huggingface/transfer-learning-conv-ai/blob/master/train.py
#   Generate: https://huggingface.co/docs/transformers/master/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate
#             https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py#L649
#   GPT2 model: https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
class DecoderAlgorithm(Algorithm):
    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.generation_config = config.generation_config

    def process_batch(self, batch):
        X, y_true, metadata = batch

        # TODO: prepare X for both LM and generation

        X = self.model.transform_inputs(X,y_true)
        y_true = self.model.transform_outputs(X, y_true)

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        X['lm_labels'] = y_true
        lm_logits, loss, y_true = self.model(**X)

        if self.is_training:
            outputs = self.model.decode_logits(lm_logits)
        else:
            outputs = self.model.generate(X['input_ids'], **self.generation_config)

        if 'labels' in metadata:
            # temporarily keep these separate until we know all labels are correctly aligned
            maybe_y_true = torch.tensor([metadata['labels'].index(y) if y in metadata['labels'] else -1 for y in y_true])
            assert(all(maybe_y_true != -1)),str(y_true)
            y_true = maybe_y_true
            outputs = torch.tensor([metadata['labels'].index(y) if y in metadata['labels'] else -1 for y in outputs])

        results = {
            'y_pred': outputs,
            'y_true': y_true,
            'metadata': metadata,
            "objective": {
                "loss_name": metadata['loss'],
                "loss_value": loss.item()}
        }

        return results, loss

    def requires_metric_calculation(self):
        if self.is_training:
            return False
        return True