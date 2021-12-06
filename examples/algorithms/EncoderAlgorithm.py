from utils import move_to
from losses import initialize_loss
from .algorithm import Algorithm

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
        super().__init__(config, datasets)

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

        # ALON LEFT OFF HERE
        # TODO: Do span extraction in EncoderAlgorithm + BertModel (needs a new layer type)

        X, y_true, metadata = getattr(self, f"_{metadata['task_metadata']['type']}_preprocessing")(X, y_true, metadata)

        X = self.model.transform_inputs(X)
        y_true = self.model.transform_outputs(y_true, metadata['task'], metadata['dataset_name'])

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        outputs = self.model(X,metadata['task'],metadata['dataset_name'])

        y_pred = getattr(self, f"_{metadata['task_metadata']['type']}_postprocessing")(outputs, y_true, metadata)

        results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'metadata': metadata,
            "objective": {"loss_name": metadata['loss']}
        }

        # the below is the original way to get the loss,
        #   unsure of if we will have any variation,
        #   or if we can just define it explicitly as cross entropy
        # metric = initialize_loss(results['objective']['loss_name'])
        metric = initialize_loss("cross_entropy")
        loss = metric.compute(outputs, y_true, return_dict=False)


        results['objective']['loss_value'] = loss.item()

        return results, loss

    def requires_metric_calculation(self):
        return True

    def _classification_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata

    def _classification_postprocessing(self, outputs, y_true, metadata):
        y_pred = multiclass_logits_to_pred(outputs)
        return y_pred

    def _span_extraction_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata

    def _span_extraction_postprocessing(self, outputs, y_true, metadata):
        pass