from utils import move_to
from losses import initialize_loss
from examples.utils import concat_t_d
from .algorithm import Algorithm
from torch import sigmoid

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
        transformed_y_true = self.model.transform_outputs(X, y_true, task_type, metadata)

        X = move_to(X, self.device)
        transformed_y_true = move_to(transformed_y_true, self.device)

        outputs = self.model(X,metadata['task'],metadata['dataset_name'])

        # task-specific loss calculation
        loss = getattr(self, f"_calculate_{task_type}_loss")(outputs, transformed_y_true, metadata, return_dict=False)

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
        X = [self.replace_sep_token(x) for x in X]
        return X, y_true, metadata

    def _classification_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        y_pred = multiclass_logits_to_pred(outputs)
        return y_pred, transformed_y_true, metadata

    def _calculate_classification_loss(self, outputs, y_true, metadata, return_dict=False):
        metric = initialize_loss("cross_entropy")
        loss = metric.compute(outputs, y_true, return_dict=return_dict)
        return loss

    def _multioutput_classification_preprocessing(self, X, y_true, metadata):
        X = [self.replace_sep_token(x) for x in X]
        return X, y_true, metadata

    def _multioutput_classification_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        # first, flatten outputs, then calculate predictions, then reshape
        outputs = outputs.reshape([-1, metadata['task_metadata']['num_labels']])
        y_pred = multiclass_logits_to_pred(outputs)
        y_pred = y_pred.reshape(transformed_y_true.shape)
        return y_pred, transformed_y_true, metadata

    def _calculate_multioutput_classification_loss(self, outputs, y_true, metadata, return_dict=False):
        # flatten the outputs and targets
        metric = initialize_loss("cross_entropy")
        outputs = outputs.reshape([-1, metadata['task_metadata']['num_labels']])
        y_true = y_true.flatten()
        loss = metric.compute(outputs, y_true, return_dict=return_dict)
        return loss

    def _multilabel_classification_preprocessing(self, X, y_true, metadata):
        X = [self.replace_sep_token(x) for x in X]
        return X, y_true, metadata

    def _multilabel_classification_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        # transform logits with sigmoid, then use a simple threshold of 0.5 for deciding output classes
        y_pred = sigmoid(outputs)
        # y_pred = (y_pred > 0.5).float()
        return y_pred, transformed_y_true, metadata

    def _calculate_multilabel_classification_loss(self, outputs, y_true, metadata, return_dict=False):
        metric = initialize_loss("BCE_with_logits")
        loss = metric.compute(outputs, y_true, return_dict=return_dict)
        return loss

    def _span_extraction_preprocessing(self, X, y_true, metadata):
        # make the tokenizer return a token offset mapping
        metadata['return_offsets_mapping'] = True
        return X, y_true, metadata

    def _span_extraction_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        del metadata['return_offsets_mapping']
        assert outputs.dim() == 3

        # decode the predictions
        y_pred_tokens = []
        pred_positions = outputs.argmax(1).tolist()
        for input_ids, (start_pred, end_pred) in zip(X.input_ids, pred_positions):
            y_pred_tokens.append(input_ids[start_pred:end_pred+1])
        y_pred = self.model.tokenizer.batch_decode(y_pred_tokens, skip_special_tokens=True)
        
        # tokenize the ground truth
        if isinstance(y_true[0], list):
            tokenized_y_true = []
            for answers in y_true:
                tokenized_answers = self.model.tokenizer([a['text'] for a in answers])
                tokenized_y_true.append(self.model.tokenizer.batch_decode(tokenized_answers.input_ids, skip_special_tokens=True))

        else:
            tokenized_answers = self.model.tokenizer([a['text'] for a in y_true])
            tokenized_y_true = self.model.tokenizer.batch_decode(tokenized_answers.input_ids, skip_special_tokens=True)

        return y_pred, tokenized_y_true, metadata

    def _calculate_span_extraction_loss(self, outputs, y_true, metadata, return_dict=False):
        start_logits, end_logits = outputs.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        ignored_index = start_logits.size(1)
        start_positions,end_positions=y_true
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        metric = initialize_loss("cross_entropy")
        start_loss = metric.compute(start_logits, start_positions, return_dict=return_dict)
        end_loss = metric.compute(end_logits, end_positions, return_dict=return_dict)
        loss = (start_loss + end_loss)/2

        return loss

    def _multiple_choice_preprocessing(self, X, y_true, metadata):
        # needs to unsqueeze the inputs
        X = [self.replace_sep_token(x) for q in X for x in q]

        # keep outputs as indices
        y_true = [int(y) for y in y_true]
        return X, y_true, metadata

    def _multiple_choice_postprocessing(self, X, outputs, y_true, transformed_y_true, metadata):
        # needs to group the data back together to make a single prediction
        outputs = outputs.view(-1, metadata['task_metadata']['num_choices'])
        y_pred = multiclass_logits_to_pred(outputs)
        return y_pred, transformed_y_true, metadata

    def _calculate_multiple_choice_loss(self, outputs, y_true, metadata, return_dict=False):
        outputs = outputs.view(-1, metadata['task_metadata']['num_choices'])
        metric = initialize_loss("cross_entropy")
        loss = metric.compute(outputs, y_true, return_dict=return_dict)
        return loss

    def replace_sep_token(self, string):
        return string.replace("[SEP]",self.model.tokenizer.sep_token)