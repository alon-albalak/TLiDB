import os
import sys
import torch
import torch.nn as nn

from utils import move_to, detach_and_clone
from optimizers import initialize_optimizer
from losses import initialize_loss

class Algorithm(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.device = config.device
        self.out_device = 'cpu'
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
        y_true = self.model.transform_outputs(y_true, metadata['task'], metadata['dataset_name'])

        X = move_to(X, self.device)
        y_true = move_to(y_true, self.device)

        if self.model.requires_y_true:
            outputs = self.model(X,metadata['task'],metadata['dataset_name'],y_true)
        else:
            outputs = self.model(X,metadata['task'],metadata['dataset_name'])

        results = {
            'y_pred': outputs,
            'y_true': y_true,
            'metadata': metadata,
            "objective": {"loss_name": metadata['loss']}
        }
        return results

    def objective(self, results, metric):
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
        metric = initialize_loss(results['objective']['loss_name'])
        objective = self.objective(results, metric)
        results['objective']['loss_value'] = objective.item()

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
        metric = initialize_loss(results['objective']['loss_name'])
        objective = self.objective(results, metric)
        results['objective']['loss_value'] = objective.item()
        return self.sanitize_dict(results)

    def train(self, mode=True):
        """
        Set the model to training mode
        """
        self.is_training = mode
        super().train(mode)

    def eval(self):
        """
        Set the model to evaluation mode
        """
        self.train(False)

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