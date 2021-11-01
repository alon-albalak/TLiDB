from torch.optim import SGD, Adam
from transformers import AdamW

def initialize_optimizer(config, model):
    # initialize optimizers
    lr = config.learning_rate
    if config.optimizer=='SGD':
        params = model.parameters()
        optimizer = SGD(
            params,
            lr=lr,
            weight_decay=config.weight_decay)
    elif config.optimizer=='AdamW':
        #TODO: this has not been tested
        if 'bert' in config.model or 'gpt' in config.model:
            no_decay = ['bias', 'LayerNorm.weight']
        else:
            no_decay = []

        # SPECIFICALLY THIS LINE
        params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            params,
            lr=lr)
    elif config.optimizer == 'Adam':
        params = model.parameters()
        optimizer = Adam(
            params,
            lr=lr,
            weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized.')

    return optimizer