import sys
import os
package_directory = os.path.dirname(os.path.abspath(__file__))
TLiDB_FOLDER = os.path.join(package_directory, "..")
sys.path.append(TLiDB_FOLDER)
import torch.nn as nn
from TLiDB.metrics.loss import ElementwiseLoss


class LM_CrossEntropyLoss:
    def __init__(self, reduction="none", ignore_index=-100):
        self.reduction = reduction
        self.ignore_index = ignore_index
    def __call__(self, output, target):
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        return nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)(output, target)

def initialize_loss(loss_function):
    if loss_function == "cross_entropy":
        return ElementwiseLoss(loss_fn = nn.CrossEntropyLoss(reduction="none"))
    elif loss_function == "LM_cross_entropy":
        return ElementwiseLoss(loss_fn = LM_CrossEntropyLoss(ignore_index=-100))
    elif loss_function == "BCE_with_logits":
        return ElementwiseLoss(loss_fn = nn.BCEWithLogitsLoss(reduction="none"))
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
