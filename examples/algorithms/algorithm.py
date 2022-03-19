import torch
import torch.nn as nn

from examples.utils import move_to, detach_and_clone
from examples.optimizers import initialize_optimizer
from examples.models.initializer import initialize_model

class Algorithm(nn.Module):
    def __init__(self, config, datasets):
        super().__init__()
        self.model = initialize_model(config, datasets)
        self.model.to(config.device)
        self.device = config.device
        self.out_device = 'cpu'
        self.optimizer = initialize_optimizer(config, self.model)
        self.max_grad_norm = config.max_grad_norm
        self.gradient_accumulation_steps = max(config.effective_batch_size//config.gpu_batch_size,1)
        self.imbalanced_task_weighting = config.imbalanced_task_weighting
        if not (config.device == 'cpu') and config.fp16:
            self.fp16 = config.fp16
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False
    
    def process_batch(self, batch):
        raise NotImplementedError

    @property
    def requires_metric_calculation(self):
        """Whether to calculate metrics"""
        return NotImplementedError
    
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load the state dict of the model
        """
        self.model.load_state_dict(state_dict)

    def update(self, batch, step):
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
        
        if self.fp16:
            with torch.cuda.amp.autocast():
                results, objective = self.process_batch(batch)
                if self.imbalanced_task_weighting:
                    task_weight = torch.tensor(batch[2]['task_weight']).to(self.device)
                    objective = task_weight * objective
                objective = objective / self.gradient_accumulation_steps
            self.scaler.scale(objective).backward()
            if ((step+1)%self.gradient_accumulation_steps) == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad(set_to_none=True)
        else:
            results, objective = self.process_batch(batch)
            if self.imbalanced_task_weighting:
                task_weight = torch.tensor(batch[2]['task_weight']).to(self.device)
                objective = task_weight * objective
            objective = objective / self.gradient_accumulation_steps
            objective.backward()
            if ((step+1)%self.gradient_accumulation_steps) == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

        return self.sanitize_dict(results)

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
        with torch.no_grad():
            results, _ = self.process_batch(batch)
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

    def convert_strings_to_labels(self, labels, strings):
        return torch.tensor([labels.index(s) if s in labels else -1 for s in strings])