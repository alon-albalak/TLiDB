from utils import move_to
from .algorithm import Algorithm
import torch


class Seq2SeqAlgorithm(Algorithm):
    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.generation_config = config.generation_config

    def process_batch(self, batch):
        X, y_true, metadata = batch
        X = self.model.transform_inputs(X)
        y_true = self.model.transform_outputs(y_true)

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
        return True