import torch.nn as nn

def initialize_loss(loss_function):
    if loss_function == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_function == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")