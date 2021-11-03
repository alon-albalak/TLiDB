import torch
from algorithms.algorithm import Algorithm
from models import initialize_model

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM) algorithm
    Minimizes the average loss of the model over all datasets
    """
    def __init__(self, config, datasets):
        model = initialize_model(config, datasets)
        model.to(config.device)
        super().__init__(config, model)
        
    def objective(self, results, metric):
        return metric.compute(results['y_pred'], results['y_true'], return_dict=False)