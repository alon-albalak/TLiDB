import tlidb.TLiDB
# dataset url can be found in the google drive, where the original link is:
#   https://drive.google.com/file/d/1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq/view?usp=sharing
# and needs to be reformatted as:
#   https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq

def get_dataset(dataset, task, dataset_folder, **dataset_kwargs):
    """
    Returns the appropriate TLiDB dataset class
    Args:
        dataset: name of the dataset
        dataset_kwargs: keyword arguments to pass to the dataset class
    Returns:
        dataset: TLiDB dataset class
    """
    if dataset not in tlidb.TLiDB.supported_datasets:
        raise ValueError(f"{dataset} is not a supported dataset, must be one of: {tlidb.TLiDB.supported_datasets}")

    if dataset == "DailyDialog":
        from tlidb.TLiDB.datasets.DailyDialog_dataset import DailyDialog_dataset as dataset_class
    elif dataset == "Friends":
        from tlidb.TLiDB.datasets.Friends_dataset import Friends_dataset as dataset_class
    elif dataset == "clinc150":
        from tlidb.TLiDB.datasets.clinc150_dataset import clinc150_dataset as dataset_class
    else:
        raise ValueError(f"{dataset} is not a supported dataset, must be one of: {tlidb.TLiDB.supported_datasets}")
        
    return dataset_class(task, dataset_folder, **dataset_kwargs)