# Data Examples

This directory holds all the code used for data handling, including:
1. Dataset loading and preprocessing
2. Data loaders
3. Metrics

If you plan to utilize the datasets with your own training script, we provide some dataloading samples below.

## Data Loading
TLiDB offers a simple, unified interface for loading datasets.

For a single dataset/task, the following example shows how to load the data, and put the data into a dataloader:


```python3
from tlidb.TLiDB.datasets.get_dataset import get_dataset
from tlidb.TLiDB.data_loaders.data_loaders import get_loader

# load the dataset, and download if necessary
dataset = get_dataset(
    dataset='DailyDialog',
    task='emotion_recognition',
    dataset_folder='TLiDB/data',
    model_type='Encoder', #Options=['Encoder', 'Decoder','EncoderDecoder']
    split='train',#Options=['train', 'dev', 'test']
    )

# get the dataloader
dataloader = get_data_loader(
    split='train', 
    dataset=dataset,
    batch_size=32,
    model_type='Encoder'
    )

# train loop
for batch in dataloader:
    X, y, metadata = batch
    ...
```

For training on multiple datasets/tasks simultaneously, TLiDB has a convenience dataloader class, TLiDB_DataLoader, which can be used to join multiple dataloaders:

```python3
from tlidb.TLiDB.datasets.get_dataset import get_dataset
from tlidb.TLiDB.data_loaders.data_loaders import get_loader, TLiDB_DataLoader

# Load the datasets, and download if necessary
source_dataset = get_dataset(
    dataset='DailyDialog',
    task='emotion_recognition',
    dataset_folder='TLiDB/data',
    model_type='Encoder', #Options=['Encoder', 'Decoder','EncoderDecoder']
    split='train',#Options=['train', 'dev', 'test']
    )

target_dataset = get_dataset(
    dataset='DailyDialog',
    task='reading_comprehension',
    dataset_folder='TLiDB/data',
    model_type='Encoder', #Options=['Encoder', 'Decoder','EncoderDecoder']
    split='train',#Options=['train', 'dev', 'test']
    )

# Get the dataloaders
source_dataloader = get_data_loader(
    split='train', 
    dataset=source_dataset,
    batch_size=32,
    model_type='Encoder'
    )
target_dataloader = get_data_loader(
    split='train', 
    dataset=target_dataset,
    batch_size=32,
    model_type='Encoder'
    )

dataloader_dict = {
    "datasets": [source_dataset, target_dataset],
    "loaders": [source_dataloader, target_dataloader],
}

# Wrap the dataloaders into a TLiDB_DataLoader
dataloader = TLiDB_DataLoader(dataloader_dict)

# train loop
for batch in dataloader:
    X, y, metadata = batch
    ...

```