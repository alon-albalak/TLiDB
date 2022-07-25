import torch
import torch.nn as nn

from tlidb.examples.utils import move_to, detach_and_clone
from tlidb.examples.optimizers import initialize_optimizer
from tlidb.examples.models.initializer import initialize_model

from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed

class Algorithm(nn.Module):
    def __init__(self, config, datasets):
        super().__init__()
        self.device = config.device
        self.out_device = 'cpu'
        self.model = initialize_model(config, datasets)
        self.optimizer = initialize_optimizer(config, self.model)
        self.deepspeed = config.deepspeed

        # TODO:
        #   TEST: MULTI-GPU, config as file path, bf16 (T5), dschf required?, 
        if self.deepspeed:
            ds_config = {
                "local_rank": config.local_rank,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": config.learning_rate,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    }
                },
                # "scheduler": {
                # "type": "WarmupDecayLR",
                #     "params": {
                #         "last_batch_iteration": -1,
                #         "total_num_steps": sum([len(loader) for loader in datasets['train']]),
                #         "warmup_min_lr": 0,
                #         "warmup_max_lr": config.learning_rate,
                #         "warmup_num_steps": sum([len(loader) for loader in datasets['train']])/100
                #     }
                # },
                "bf16": {
                    "enabled": False
                },
                "fp16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 2,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "allgather_bucket_size":5e8,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e8,
                },
                "gradient_clipping": 1.0,
                "steps_per_print": 500,
                "train_batch_size": config.effective_batch_size,
                "train_micro_batch_size_per_gpu": config.gpu_batch_size,
                "gradient_accumulation": config.effective_batch_size // config.gpu_batch_size,
                "wall_clock_breakdown": False
            }
            dschf = HfDeepSpeedConfig(ds_config)
            self.model.model, self.optimizer, _, _ = deepspeed.initialize(model=self.model.model, config_params=ds_config, optimizer=self.optimizer)
        else:
            self.model.to(config.device)
            self.max_grad_norm = config.max_grad_norm
            self.gradient_accumulation_steps = max(config.effective_batch_size//config.gpu_batch_size,1)
        
        self.imbalanced_task_weighting = config.imbalanced_task_weighting
        if not (config.device == 'cpu') and config.fp16:
            self.fp16 = config.fp16
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False
        if config.pipeline_parallel:
            self.model.parallelize()
    
    def process_batch(self, batch):
        raise NotImplementedError

    @property
    def requires_metric_calculation(self):
        """Whether to calculate metrics"""
        return NotImplementedError
    
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

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
        
        if self.deepspeed:
            results, objective = self.process_batch(batch)
            if self.imbalanced_task_weighting:
                task_weight = torch.tensor(batch[2]['task_weight']).to(self.device)
                objective = task_weight * objective
            self.model.backward(objective)
            self.model.step()

        elif self.fp16:
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