from utils import move_to
from .algorithm import Algorithm
from models import initialize_model
import torch


class Seq2SeqAlgorithm(Algorithm):
    def __init__(self, config, datasets):
        model = initialize_model(config, datasets)
        model.to(config.device)
        super().__init__(config, model)

    def process_batch(self, batch):
        X, y_true, metadata = batch
        X = self.model.transform_inputs(X)
        y_true = self.model.transform_outputs(y_true, metadata['task'], metadata['dataset_name'])

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        X['lm_labels'] = y_true
        outputs, loss, y_true = self.model(**X)

        if 'labels' in metadata:
            y_true = torch.tensor([metadata['labels'].index(y) if y in metadata['labels'] else -1 for y in y_true])
            assert(all(y_true != -1))
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