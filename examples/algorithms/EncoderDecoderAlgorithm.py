from utils import move_to
from .algorithm import Algorithm
import torch


class EncoderDecoderAlgorithm(Algorithm):
    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.generation_config = config.generation_config
        self.generate_during_training = config.generate_during_training

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
                - objective: a dictionary with the loss name and loss value
        """
        X, y_true, metadata = batch
        
        # task-specific preprocessing
        X, y_true, metadata = getattr(self, f"_{metadata['task_metadata']['type']}_preprocessing")(X, y_true, metadata)

        # prepare inputs for generation if necessary
        if self.requires_metric_calculation():
            self.model.tokenizer.padding_side="left"
            X_generate = self.model.transform_inputs(X).input_ids
            X_generate = move_to(X_generate, self.device)
            self.model.tokenizer.padding_side="right"

        X = self.model.transform_inputs(X)
        lm_labels = self.model.transform_outputs(y_true)

        X['lm_labels'] = lm_labels
        X = move_to(X, self.device)

        loss = self.model(**X)

        # generate predictions and convert to labels if necessary
        if self.requires_metric_calculation():
            # generate predictions
            outputs = self.model.generate(X_generate, **self.generation_config)
            
            # task-specific postprocessing
            y_pred, y_true = getattr(self, f"_{metadata['task_metadata']['type']}_postprocessing")(outputs, y_true, metadata)

            if 'labels' in metadata:
                # temporarily keep these separate until we know all labels are correctly aligned
                maybe_y_true = torch.tensor([metadata['labels'].index(y) if y in metadata['labels'] else -1 for y in y_true])
                assert(all(maybe_y_true != -1)),str(y_true)
                y_true = maybe_y_true
                y_pred = torch.tensor([metadata['labels'].index(y) if y in metadata['labels'] else -1 for y in y_pred])
        else:
            y_pred = []
            y_true = []

        results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'metadata': metadata,
            "objective": {
                "loss_name": "LM_cross_entropy",
                "loss_value": loss.item()}
        }

        return results, loss

    def requires_metric_calculation(self):
        # determines whether the model needs to generate predictions
        #       else only calculates loss
        if self.is_training and not self.generate_during_training:
            return False
        return True

    def _classification_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata

    def _classification_postprocessing(self, outputs, y_true, metadata):
        y_pred = self.model.batch_decode(outputs)
        return y_pred, y_true

    def _span_extraction_preprocessing(self, X, y_true, metadata):
        y_true = [y['text'] for y in y_true]
        return X, y_true, metadata

    def _span_extraction_postprocessing(self, outputs, y_true, metadata):
        y_pred = self.model.batch_decode(outputs)
        return y_pred, y_true

    def _multiple_choice_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata

    def _multiple_choice_postprocessing(self, outputs, y_true, metadata):
        y_pred = self.model.batch_decode(outputs)
        num_choices = metadata['task_metadata']['num_choices']
        metadata['labels'] = [str(i) for i in range(num_choices)]
        return y_pred, y_true