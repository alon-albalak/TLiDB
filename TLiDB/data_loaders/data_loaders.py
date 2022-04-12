from torch.utils.data import DataLoader
import random

def get_train_loader(dataset, batch_size, model_type, **loader_kwargs):
    """
    Constructs and return the data loader for training
    Args:
        - dataset (TLiDBDataset): The dataset to load the data from
        - batch_size (int): The batch size for the data loader
        - **loader_kwargs (dict): The keyword arguments for the data loader
    Returns:
        - data_loader (DataLoader): The data loader for training
    """
    if dataset.task_metadata['type'] == "multiple_choice" and model_type == "Encoder":
        # Encoder-only models split multiple choice into num_choices samples
        #   so we need to downscale the batch_size accordingly
        batch_size = batch_size // dataset.task_metadata['num_choices']
    if loader_kwargs['num_workers'] > 1:
        loader_kwargs['pin_memory'] = True
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

def get_eval_loader(dataset, batch_size, model_type, **loader_kwargs):
    """
    Constructs and return the data loader for evaluation
    Args:
        - dataset (TLiDBDataset): The dataset to load the data from
        - batch_size (int): The batch size for the data loader
        - **loader_kwargs (dict): The keyword arguments for the data loader
    Returns:
        - data_loader (DataLoader): The data loader for evaluation
    """
    if dataset.task_metadata['type'] == "multiple_choice" and model_type == "Encoder":
        # Encoder-only models split multiple choice into num_choices samples
        #   so we need to downscale the batch_size accordingly
        batch_size = batch_size // dataset.task_metadata['num_choices']
    if loader_kwargs['num_workers'] > 1:
        loader_kwargs['pin_memory'] = True
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

def get_dataloader(split, dataset, batch_size, model_type, **loader_kwargs):
    if split == 'train':
        return get_train_loader(dataset, batch_size, model_type, **loader_kwargs)
    else:
        return get_eval_loader(dataset, batch_size, model_type, **loader_kwargs)

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
        self.lengths = [len(d) for d in self.dataloaders]
        self.remaining_batches = [len(d) for d in self.dataloaders]
        self.task_metadatas = [d.task_metadata for d in dataloader_dict['datasets']]
        # inverse square-root weights are stored in metadata, if desired
        self.task_weights = [1/l**(1/2) for l in self.lengths]

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
            metadata['task_metadata'] = self.task_metadatas[selected_idx]
            metadata['task_weight'] = self.task_weights[selected_idx]
            return X,y,metadata
        else:
            raise StopIteration