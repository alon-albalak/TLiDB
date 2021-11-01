import torch
from examples.models.initializer import initialize_model
from examples.algorithms.algorithm import Algorithm

class ERM(Algorithm):
    def __init__(self, config, loss, metric, dataset):
        model = initialize_model(config, dataset)
        model.to(config.device)
        super().__init__(config, model, loss, metric)
        
    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)