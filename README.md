# The Transfer Learning in Dialogue Benchmarking Toolkit
[![PyPI](https://img.shields.io/pypi/v/tlidb)](https://pypi.org/project/tlidb/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alon-albalak/tlidb/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/419109889.svg)](https://zenodo.org/badge/latestdoi/419109889)

## Overview
TLiDB is a tool used to benchmark methods of transfer learning in conversational AI.
TLiDB can easily handle *domain adaptation, task transfer, multitasking, continual learning*, and other transfer learning settings.
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
 - sentencepiece>=0.1.96
 - bert-score==0.3.11


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
TLiDB can be used from the command line or as a python command. If you have installed the package from source, we highly recommend running commands from inside the tlidb/examples/ directory.

### Quick Start
For a very simple set up, you can use the following commands.
- From command line:
```bash
tlidb --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks reading_comprehension --do_train --do_finetune --do_eval --eval_best --model_config=bert
```
- As python command (only if installed from source):
```bash
cd examples
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks reading_comprehension --do_train --do_finetune --do_eval --eval_best --model_config=bert
```

### Detailed Usage

TLiDB has 2 main folders of interest:
- `tlidb/examples`
- `tlidb/TLiDB`

`tlidb/examples/` is recommended for use if you would like to utilize our training scripts. It contains sample code for models, learning algorithms, and sample training scripts. 
For detailed examples, see the [Examples README](/tlidb/examples/README.md).

`tlidb/TLiDB/` holds the code related to data (datasets, dataloaders, metrics, etc.). If you are interested in utilizing our datasets and metrics but would like to train models using your own training scripts, take a look at the example usage in [TLiDB README](/tlidb/TLiDB/README.md).


## Folder descriptions:
- tlidb/TLiDB is the folder holding the code for data handling
    - tlidb/TLiDB/data_loaders contains code for data_loaders
    - tlidb/TLiDB/data is the destination folder for downloaded datasets (if installed from source, otherwise data is in .cache/tlidb/data)
    - tlidb/TLiDB/datasets contains code for dataset loading and preprocessing
    - tlidb/TLiDB/metrics contains code for loss and evaluation metrics
    - tlidb/TLiDB/utils contains utility files
- tlidb/examples contains sample code for training and evaluating models
    - tlidb/examples/algorithms contains code which trains and evaluates a model
    - tlidb/examples/models contains code to define a model
    - tlidb/examples/configs contains code for model configurations
- /dataset_preprocessing is for reproducability purposes. It contains scripts used to preprocess the TLiDB datasets from their original form into the standardized TLiDB format

## Comments, Questions, and Feedback
If you find issues, please [open an issue here](https://github.com/alon-albalak/TLiDB/issues).

If you have dataset or model requests, please [add a new discussion here](https://github.com/alon-albalak/TLiDB/discussions).

We encourage outside contributions to the project!



## Citation
If you use the FETA datasets in your work, please cite the FETA paper:
```
@misc{feta_albalak,
    doi = {10.48550/ARXIV.2205.06262},
    url = {https://arxiv.org/abs/2205.06262},
    author = {Albalak, Alon and Tuan, Yi-Lin and Jandaghi, Pegah and Pryor, Connor and Yoffe, Luke and Ramachandran, Deepak and Getoor, Lise and Pujara, Jay and Wang, William Yang},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {FETA: A Benchmark for Few-Sample Task Transfer in Open-Domain Dialogue},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution Share Alike 4.0 International}
    }
```

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
