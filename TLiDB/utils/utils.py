import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)

def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Setting seed to {seed}")

def numel(obj):
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError("Invalid type for numel")
