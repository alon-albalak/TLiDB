import sys
import os
package_directory = os.path.dirname(os.path.abspath(__file__))
TLiDB_FOLDER = os.path.join(package_directory, "..")
sys.path.append(TLiDB_FOLDER)
import torch.nn as nn
from TLiDB.utils.metrics.loss import ElementwiseLoss

def initialize_loss(loss_function):
    if loss_function == "cross_entropy":
        return ElementwiseLoss(loss_fn = nn.CrossEntropyLoss(reduction="none"))
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
