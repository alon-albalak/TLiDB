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
        task_type=metadata['task_metadata']['type']

        # task-specific preprocessing
        X, y_true, metadata = getattr(self, f"_{task_type}_preprocessing")(X, y_true, metadata)

        X = self.model.transform_inputs(X, metadata)
        transformed_y_true = self.model.transform_outputs(X, y_true, task_type, metadata['task'], metadata['dataset_name'])

        X = move_to(X, self.device)
        transformed_y_true = move_to(transformed_y_true, self.device)

        outputs = self.model(X,metadata['task'],metadata['dataset_name'])

        # task-specific loss calculation
        loss = getattr(self, f"_calculate_{task_type}_loss")(outputs, transformed_y_true, return_dict=False)

        # task-specific postprocessing
        y_pred, y_true, metadata = getattr(self, f"_{task_type}_postprocessing")(X, outputs, y_true, transformed_y_true, metadata)

        results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'metadata': metadata,
            "objective": {"loss_name": "cross_entropy"}
        }

        results['objective']['loss_value'] = loss.item()

        return results, loss

    def requires_metric_calculation(self):
        return True

    def _classification_preprocessing(self, X, y_true, metadata):
        metadata['return_offsets_mapping'] = False
        return X, y_true, metadata

    def _classification_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        y_pred = multiclass_logits_to_pred(outputs)
        del metadata['return_offsets_mapping']
        return y_pred, transformed_y_true, metadata

    def _calculate_classification_loss(self, outputs, y_true, return_dict=True):
        metric = initialize_loss("cross_entropy")
        loss = metric.compute(outputs, y_true, return_dict=False)
        return loss

    def _span_extraction_preprocessing(self, X, y_true, metadata):
        # make the tokenizer return a token offset mapping
        metadata['return_offsets_mapping'] = True
        return X, y_true, metadata

    def _span_extraction_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        del metadata['return_offsets_mapping']
        assert outputs.dim() == 3
        y_pred_tokens = []

        pred_positions = outputs.argmax(1).tolist()
        for input_ids, (start_pred, end_pred) in zip(X.input_ids, pred_positions):
            y_pred_tokens.append(input_ids[start_pred:end_pred+1])
        y_pred = self.model.tokenizer.batch_decode(y_pred_tokens, skip_special_tokens=True)
        y_true = [y['text'] for y in y_true]
        return y_pred, y_true, metadata

    def _calculate_span_extraction_loss(self, outputs, y_true, return_dict=True):
        start_logits, end_logits = outputs.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        ignored_index = start_logits.size(1)
        start_positions,end_positions=y_true
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        metric = initialize_loss("cross_entropy")
        start_loss = metric.compute(start_logits, start_positions, return_dict=False)
        end_loss = metric.compute(end_logits, end_positions, return_dict=False)
        loss = (start_loss + end_loss)/2

        return loss