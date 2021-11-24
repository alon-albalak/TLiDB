import torch.nn as nn

from utils import move_to
from optimizers import initialize_optimizer
from losses import initialize_loss
from .algorithm import Algorithm
from models import initialize_model

# temporarily here
# TODO this function needs a better home
def multiclass_logits_to_pred(logits):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1
    return logits.argmax(-1)

class EncoderAlgorithm(Algorithm):
    def __init__(self, config, datasets):
        model = initialize_model(config, datasets)
        model.to(config.device)
        super().__init__(config, model)

    def process_batch(self, batch):
        """
        A helper function for update() and evaluate() that process the batch
        Args:
            - batch: a batch of data yielded by the DataLoader
        Output:
            - results: a dictionary of results
                - y_pred: the prediction of the model
                - y_true: the ground truth
                - metadata: the metadata of the batch
        """
        X, y_true, metadata = batch
        X = self.model.transform_inputs(X)
        y_true = self.model.transform_outputs(y_true, metadata['task'], metadata['dataset_name'])

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        outputs = self.model(X,metadata['task'],metadata['dataset_name'])

        preds = multiclass_logits_to_pred(outputs)
        results = {
            'y_pred': preds,
            'y_true': y_true,
            'metadata': metadata,
            "objective": {"loss_name": metadata['loss']}
        }

        metric = initialize_loss(results['objective']['loss_name'])
        loss = metric.compute(outputs, y_true, return_dict=False)


        results['objective']['loss_value'] = loss.item()

        return results, loss

    # def evaluate(self, batch):
    #     """
    #     Process the batch, and evaluate the model
    #     Args:
    #         - batch: a batch of data yielded by data loaders
    #     Output:
    #         - results (dict): information about the batch, such as:
    #             - y_pred: the predicted labels
    #             - y_true: the true labels
    #             - metadata: the metadata of the batch
    #             - loss: the loss of the batch
    #             - metrics: the metrics of the batch
    #     """
    #     assert not self.is_training, "Cannot evaluate() when in training mode"
    #     results, _ = self.process_batch(batch)
    #     return self.sanitize_dict(results)
