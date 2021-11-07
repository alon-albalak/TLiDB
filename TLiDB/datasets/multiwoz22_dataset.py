from .TLiDB_dataset import TLiDB_Dataset

class multiwoz22_dataset(TLiDB_Dataset):
    _dataset_name = "multiwoz22"
    _tasks = ["dialogue_state_tracking","dialogue_response_generation"]
    _url= "https://drive.google.com/uc?export=download&id=1ZYiKM6D2-b8HfIP_jHh6-YdtJ6sZfssQ"
    def __init__(self, task, dataset_folder):
        assert task in self.tasks, f"{task} is not a valid task for {self.dataset_name}"
        super().__init__(self.dataset_name, task, dataset_folder=dataset_folder)