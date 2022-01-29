
import TLiDB
DATASET_FOLDER=TLiDB.TLiDB_path+"/data/"
# dataset url can be found in the google drive, where the original link is:
#   https://drive.google.com/file/d/1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq/view?usp=sharing
# and needs to be reformatted as:
#   https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq
# DATASETS_INFO = {
#     "friends_RC": {"dataset_class": friends_RC_dataset,
#                    "url": "https://drive.google.com/uc?export=download&id=1Gi70GnNNRQWgnJNaOpbx9vVKq7L8Lbte"},
#     "friends_QA": {"dataset_class": friends_QA_dataset,
#                    "url": "https://drive.google.com/uc?export=download&id=1WlpmRNoYW5zXrOqBNw0OhVjyZ-eFXMCm"},
#     "banking77": {"dataset_class": banking77_datset,
#                    "url": "https://drive.google.com/uc?export=download&id=AUHAh8czIyhR9FbBdHi1oOt7ZtgSNl8W"},
# }
def get_dataset(dataset, task, dataset_folder=DATASET_FOLDER, **dataset_kwargs):
    """
    Returns the appropriate TLiDB dataset class
    Args:
        dataset: name of the dataset
        dataset_kwargs: keyword arguments to pass to the dataset class
    Returns:
        dataset: TLiDB dataset class
    """
    if dataset not in TLiDB.supported_datasets:
        raise ValueError(f"{dataset} is not a supported dataset, must be one of: {TLiDB.supported_datasets}")

    if dataset == "DailyDialog":
        from TLiDB.datasets.DailyDialog_dataset import DailyDialog_dataset as dataset_class
    elif dataset == "Friends":
        from TLiDB.datasets.Friends_dataset import Friends_dataset as dataset_class
    elif dataset == "clinc150":
        from TLiDB.datasets.clinc150_dataset import clinc150_dataset as dataset_class
    elif dataset == "friends_ER":
        from TLiDB.datasets.friends_ER_dataset import friends_ER_dataset as dataset_class
        
    return dataset_class(task, dataset_folder, **dataset_kwargs)