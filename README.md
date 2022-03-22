# The Transfer Learning in Dialogue Benchmarking Toolkit
[![PyPI](https://img.shields.io/pypi/v/tlidb)](https://pypi.org/project/tlidb/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alon-albalak/tlidb/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/419109889.svg)](https://zenodo.org/badge/latestdoi/419109889)

## Overview
TLiDB is a tool used to benchmark methods of transfer learning in conversational AI.
TLiDB can easily handle domain adaptation, task transfer, multitasking, continual learning, and other transfer learning settings.
TLiDB maintains a unified json format for all datasets and tasks, easing the new code necessary for new datasets and tasks. We highly encourage community contributions to the project.

The main features of TLiDB are:

1. Dataset class to easily load a dataset for use across models
2. Unified metrics to standardize evaluation across datasets
3. Extensible Model and Algorithm classes to support fast prototyping

## Installation

#### Requirements
 - python>=3.6
 - torch>=1.10
 - nltk>=3.6.5
 - scikit-learn>=1.0
 - transformers>=4.11.3
 - sentencepiece>=0.1.96 (optional)


To use TLiDB, you can simply install via `pip`:
```bash
pip install tlidb
```

OR, you can install TLiDB from source. This is recommended if you want to edit or contribute:
```bash
git clone git@github.com:alon-albalak/TLiDB.git
cd TLiDB
pip install -e .
```

## How to use TLiDB
TLiDB has 2 main folders of interest:
- `/TLiDB`
- `/examples`

`/TLiDB/` holds the code related to data (datasets, dataloaders, metrics, etc.)

`/examples/` contains sample code for models, learning algorithms, and sample training scripts. 
For detailed examples, see the [Examples README](/examples/README.md).

### Data Loading
TLiDB offers a simple, unified interface for loading datasets.

For a single dataset/task, the following example shows how to load the data, and put the data into a dataloader:


```python3
from TLiDB.datasets.get_dataset import get_dataset
from TLiDB.data_loaders.data_loaders import get_loader

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
from TLiDB.datasets.get_dataset import get_dataset
from TLiDB.data_loaders.data_loaders import get_loader, TLiDB_DataLoader

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


## Folder descriptions:
- /TLiDB is the main folder holding the code for data
    - /TLiDB/data_loaders contains code for data_loaders
    - /TLiDB/data is the destination folder for downloaded datasets
    - /TLiDB/datasets contains code for datasets
    - /TLiDB/metrics contains code for loss and evaluation metrics
    - /TLiDB/utils contains utility files
- /examples contains sample code for training models
    - /examples/algorithms contains code which trains and evaluates a model
    - /examples/models contains code to define a model
    - /examples/configs contains code for model configurations
    - /examples/logs_and_models is the default destination folder for training logs and model checkpoints
- /dataset_preprocessing is for reproducability purposes. It contains scripts used to preprocess the TLiDB datasets from their original form into the TLiDB form

## Citation
If you use TLiDB in your work, please cite the repository:
```
@software{Albalak_The_Transfer_Learning_2022,
author = {Albalak, Alon},
doi = {10.5281/zenodo.6374360},
month = {3},
title = {{The Transfer Learning in Dialogue Benchmarking Toolkit}},
url = {https://github.com/alon-albalak/TLiDB},
version = {1.0.0},
year = {2022}
}
```

## Acknowledgements
The design of TLiDB was based the [wilds](https://github.com/p-lambda/wilds) project, and the [Open Graph Benchmark](https://github.com/snap-stanford/ogb).
