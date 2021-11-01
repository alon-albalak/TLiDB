import torch
import torch.nn as nn
from examples.utils import move_to, detach_and_clone
from examples.optimizers import initialize_optimizer

class Algorithm(nn.Module):
    def __init__(self, config, model, loss, metric):
        super().__init__()
        self.device = config.device
        self.out_device = 'cpu'
        self.loss = loss
        self.optimizer = initialize_optimizer(config, model)
        self.max_grad_norm = config.max_grad_norm
        self.model = model
    
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
        y_true = self.model.transform_outputs(y_true)

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        outputs = self.model(X)

        results = {
            'y_pred': outputs,
            'y_true': y_true,
            'metadata': metadata,
        }
        return results

    def objective(self, results):
        raise NotImplementedError

    def update(self, batch):
        """
        Process the batch, and update the model
        Args:
            - batch: a batch of data yielded by data loaders
        Output:
            - results (dict): information about the batch, such as:
                - y_pred: the predicted labels
                - y_true: the true labels
                - metadata: the metadata of the batch
                - loss: the loss of the batch
                - metrics: the metrics of the batch
        """
        assert self.is_training, "Cannot update() when not in training mode"

        results = self.process_batch(batch)
        self._update(results)
        return self.sanitize_dict(results)

    def _update(self, results):
        """
        Computes the objective and updates the model
        """
        objective = self.objective(results)
        results['objective'] = objective.item()

        # update the model
        objective.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.model.zero_grad()


    def evaluate(self, batch):
        """
        Process the batch, and evaluate the model
        Args:
            - batch: a batch of data yielded by data loaders
        Output:
            - results (dict): information about the batch, such as:
                - y_pred: the predicted labels
                - y_true: the true labels
                - metadata: the metadata of the batch
                - loss: the loss of the batch
                - metrics: the metrics of the batch
        """
        assert not self.is_training, "Cannot evaluate() when in training mode"
        results = self.process_batch(batch)
        results['objective'] = self.objective(results).item()
        return self.sanitize_dict(results)

    def train(self, mode=True):
        """
        Set the model to training mode
        """
        self.is_training = mode
        super().train(mode)

    def sanitize_dict(self, in_dict, to_out_device=True):
        """
        Helper function that sanitizes dictionaries by:
            - moving to the specified output device
            - removing any gradient information
            - detaching and cloning the tensors
        Args:
            - in_dict (dictionary)
        Output:
            - out_dict (dictionary): sanitized version of in_dict
        """
        out_dict = detach_and_clone(in_dict)
        if to_out_device:
            out_dict = move_to(out_dict, self.out_device)
        return out_dict
