# The Transfer Learning in Dialogue Benchmark

### File structure:

- TLiDB/
    - /datasets/
        - sample_format.json
        - dataset1.json
        - dataset2.json
    - /utils/
        - datasets.py
        - metrics.py
        - utils.py
    - train.py
- examples/
    - model1.py
    - model2.py
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

### Folder descriptions:

- /TLiDB is the main folder holding the benchmark
- /examples contains sample code for training models
- /distances contains code for calculating distances between datasets/domains/tasks
- /dataset_preprocessing is for reproducability purposes, not required for end users. It contains scripts used to preprocess the TLiDB datasets from their original form into the TLiDB form
