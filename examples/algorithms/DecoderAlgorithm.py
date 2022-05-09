from examples.utils import move_to
from .algorithm import Algorithm

class DecoderAlgorithm(Algorithm):
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
            X_generate = self.model.transform_generation_inputs(X)
            X_generate = move_to(X_generate, self.device)

        X, lm_labels = self.model.transform_LM_inputs(X,y_true)
        X['lm_labels'] = lm_labels
        X = move_to(X, self.device)

        # track number of tokens in the batch
        num_batch_tokens = X['attention_mask'].sum().item()

        loss = self.model(**X)

        # generate predictions and convert to labels if necessary
        if self.requires_metric_calculation():
            # generate predictions
            outputs = self.model.generate(X_generate, metadata['task_metadata']['max_decode_tokens'], **self.generation_config)

            # task-specific postprocessing
            y_pred, y_true = getattr(self, f"_{metadata['task_metadata']['type']}_postprocessing")(outputs, y_true, metadata)

        else:
            y_pred = []
            y_true = []

        results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'metadata': metadata,
            'batch_loss_divisor': num_batch_tokens, # used for averaging loss
            "objective": {
                "loss_name": "LM_cross_entropy",
                "loss_value": loss.item()*num_batch_tokens}
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
        y_true = self.convert_strings_to_labels(metadata['labels'], y_true)
        assert(all(y_true != -1)),str(y_true)
        y_pred = self.convert_strings_to_labels(metadata['labels'], outputs)
        return y_pred, y_true

    def _multioutput_classification_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata

    def _multioutput_classification_postprocessing(self, outputs, y_true, metadata):
        y_true = self.convert_strings_to_labels(metadata['labels'], y_true)
        assert(all(y_true != -1)),str(y_true)
        y_pred = self.convert_strings_to_labels(metadata['labels'], outputs)
        return y_pred, y_true
    
    def _multilabel_classification_preprocessing(self, X, y_true, metadata):
        # format y_true into a string of labels
        y_true = [" and ".join([metadata['task_metadata']['class_to_natural_language_map'][c] for c in sample]) for sample in y_true]
        return X, y_true, metadata

    def _multilabel_classification_postprocessing(self, outputs, y_true, metadata):
        # convert model outputs to mutlilabel format
        y_pred = []
        for output in outputs:
            pred = [0 for _ in range(len(metadata['labels']))]
            # search for class names in output
            for i, natural_language_class in enumerate(metadata['task_metadata']['class_to_natural_language_map'].values()):
                if natural_language_class in output:
                    prediction = list(metadata['task_metadata']['class_to_natural_language_map'].keys())[i]
                    pred[i] = 1
            if sum(pred) == 0:
                pred[metadata['labels'].index(metadata['task_metadata']['default_prediction'])] = 1
            y_pred.append(pred)

        # convert labels to multilabel format
        transformed_y_true = []
        for y in y_true:
            true = [0 for _ in range(len(metadata['labels']))]
            natural_language_labels = y.split(" and ")
            label_indices = [list(metadata['task_metadata']['class_to_natural_language_map'].values()).index(l) for l in natural_language_labels]
            for i in label_indices:
                true[i] = 1
            transformed_y_true.append(true)

        return y_pred, transformed_y_true

    def _span_extraction_preprocessing(self, X, y_true, metadata):
        if isinstance(y_true[0], list):
            y_true = [[y_['text'] for y_ in y] for y in y_true]
        else:
            y_true = [y['text'] for y in y_true]
        return X, y_true, metadata

    def _span_extraction_postprocessing(self, outputs, y_true, metadata):
        y_pred = outputs
        return y_pred, y_true

    def _multiple_choice_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata

    def _multiple_choice_postprocessing(self, outputs, y_true, metadata):
        num_choices = metadata['task_metadata']['num_choices']
        metadata['labels'] = [str(i) for i in range(num_choices)]
        y_true = self.convert_strings_to_labels(metadata['labels'], y_true)
        assert(all(y_true != -1)),str(y_true)
        y_pred = self.convert_strings_to_labels(metadata['labels'], outputs)
        return y_pred, y_true

    def _response_generation_preprocessing(self, X, y_true, metadata):
        return X, y_true, metadata
    
    def _response_generation_postprocessing(self, outputs, y_true, metadata):
        y_pred = outputs
        return y_pred, y_true