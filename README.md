# The Transfer Learning in Dialogue Benchmark

### Folder descriptions:

- /TLiDB is the main folder holding the benchmark
    - /TLiDB/data_loaders contains code for data_loaders
    - /TLiDB/datasets is the destination folder for downloaded datasets
    - /TLiDB/metrics contains code for loss and evaluation metrics
    - /TLiDB/utils contains utility files (data downloader, logging, argparser, etc.)
- /examples contains sample code for training models
    - /examples/algorithms contains code which trains and evaluates a model
    - /examples/models contains code to define a model
- /distances contains code for calculating distances between datasets/domains/tasks
- /dataset_preprocessing is for reproducability purposes, not required for end users. It contains scripts used to preprocess the TLiDB datasets from their original form into the TLiDB form


### File structure:

- TLiDB/
    - /data_loaders/
        - data_loader.py
    - /datasets/
        - sample_format.json
        - dataset1.json
        - dataset2.json
    - /metrics/
        - loss.py
        - metric.py
    - /utils/
        - TLiDB_datasets.py
        - argparser.py
        - utils.py
- examples/
    - train.py
    - evaluate.py
    - run_experiment.py
    - losses.py
    - optimizers.py
    - utils.py
    - algorithms/
        - algorithm1.py
        - algorithm2.py
        - initializer.py
    - models/
        - model1.py
        - model2.py
        - initializer.py
    - configs/
        - datasets.json
        - data_loader.json
        - models.json
- distances/
    - domain_distance.py
    - task_distance.py
- dataset_preprocessing/
    - dataset1/
        - convert_dataset1.py
        - download_and_convert_dataset1.sh
    - dataset2/
        - convert_dataset2.py
        - download_and_convert_dataset2.sh
