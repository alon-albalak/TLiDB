from losses import initialize_loss
from algorithms.ERM import ERM

def initialize_algorithm(config, datasets):
    # For a list of datasets
    train_dataset = datasets['train']['dataset']
    train_loader = datasets['train']['loader']

    loss = initialize_loss(config.loss_function)
    # metric = get task-specific metrics
    metric = None

    if config.algorithm == "ERM":
        algorithm = ERM(config, loss, metric, train_dataset)


    else:
        raise ValueError(f"Invalid algorithm name: {config.algorithm}")

    return algorithm