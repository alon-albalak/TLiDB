from utils import move_to
from .algorithm import Algorithm
import torch


class Seq2SeqAlgorithm(Algorithm):
    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.generation_config = config.generation_config

    def process_batch(self, batch):
        X, y_true, metadata = batch

        # for span extraction tasks, keep outputs in tokenized form
        if 'span_extraction' in metadata['task_annotation_type']:
            output_type = "tokens"
        else:
            output_type = "string"

        X = self.model.transform_inputs(X)

        # TODO: ALON LEFT OFF HERE
        # algorithm diverges depending on task type
        #   for span extraction task, only y_true['text'] is passed
        #   pass in [y['text'] for y in y_true]]
        # ~~~~~NEED TO DO SOMETHING SIMILAR FOR ENCODERALGORITHM~~~~~~
        if metadata['task_annotation_type'] == 'span_extraction':
            y_true = [y['text'] for y in y_true]
        y_true = self.model.transform_outputs(y_true)

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        X['lm_labels'] = y_true
        lm_logits, loss, decoded_y_true = self.model(**X)

        if self.is_training:
            outputs = self.model.greedy_decode_logits(lm_logits)
        else:
            outputs = self.model.generate(X['input_ids'], **self.generation_config)

        if output_type == "string":
            outputs = self.model.batch_decode(outputs)
            y_true = decoded_y_true
        elif output_type == "tokens":
            # convert to list of lists of token ids
            outputs = outputs.cpu().tolist()
            y_true = y_true.cpu().tolist()
        else:
            raise ValueError(f"output_type {output_type} not recognized")

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