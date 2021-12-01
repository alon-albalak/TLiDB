from torch.utils.data import DataLoader
import numpy as np
import random

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


class TLiDB_DataLoader:
    """
    Data loader that combines and samples from multiple single-task data loaders

    Args:
        - dataloader_dict (dict): A dictionary containing the following keys:
            - datasets (list): A list of TLiDB_Dataset objects
            - loaders (list): A list of DataLoader objects
            - losses (list): A list of strings corresponding to the name of the loss for each dataset
    Returns:
        - batch (tuple): A tuple containing the following elements:
            - X (str): The input data
            - y (str): The target data
            - metadata (dict): A dictionary containing the following keys:
                - task (str): The name of the task
                - dataset_name (str): The name of the dataset
                - loss (str): The name of the loss
    """
    def __init__(self, dataloader_dict):
        self.dataset_names = [d.dataset_name for d in dataloader_dict['datasets']]
        self.task_names = [d.task for d in dataloader_dict['datasets']]
        self.dataloaders = [iter(d) for d in dataloader_dict['loaders']]
        self.losses = dataloader_dict['losses']
        self.lengths = [len(d) for d in self.dataloaders]
        self.remaining_batches = [len(d) for d in self.dataloaders]

    def _reset(self):
        pass

    def __iter__(self):
        self._reset()
        return self
    
    def __len__(self):
        return sum(self.lengths)

    def __next__(self):
        if sum(self.remaining_batches) > 0:
            # sample probabilistically from each data loader
            selected_idx = random.choices(range(len(self.dataloaders)), weights=self.remaining_batches)[0]
            selected_loader = self.dataloaders[selected_idx]
            self.remaining_batches[selected_idx] -= 1
            batch = next(selected_loader)
            X, y, metadata = batch
            metadata['task'] = self.task_names[selected_idx]
            metadata['dataset_name'] = self.dataset_names[selected_idx]
            metadata['loss'] = self.losses[selected_idx]
            # TODO: include label mapping if dataset has labels
            meta_fields = selected_loader._dataset.metadata_fields
            meta_array = selected_loader._dataset.metadata_array
            if 'labels' in meta_fields:
                metadata['label_mapping'] = meta_array[meta_fields.index('labels')]
            return X,y,metadata
        else:
            raise StopIteration