from torch.utils.data import DataLoader
import numpy as np

def get_train_loader(dataset, batch_size, **loader_kwargs):
    """
    Constructs and return the data loader for training
    Args:
        - dataset (TLiDBDataset): The dataset to load the data from
        - batch_size (int): The batch size for the data loader
        - **loader_kwargs (dict): The keyword arguments for the data loader
    Returns:
        - data_loader (DataLoader): The data loader for training
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)


class Multitask_DataLoader:
    """
    Data loader that combines and samples from multiple single-task data loaders
    """
    def __init__(self, dataloader_dict):
        pass