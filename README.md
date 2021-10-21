# The Transfer Learning in Dialogue Benchmark

### File structure:

The following files are for public use
- models/
    - /model1.py
    - /model2.py
- datasets/
    - /dataset1/
        - /data files
    - /dataset2/
        - /data files
- utils/
    - /dataloader.py


The following files are for internal use.

For example, we keep track of scripts used to convert the original datasets into our unified format
Please do not push dataset folders to the github repo. Instead, zip and upload them to the google drive, found at: https://drive.google.com/drive/folders/1PiMzi0GpV1QuKUazNAYAbUEmHS_M1tfr?usp=sharing
- internal_use_files/
    - /data_conversions/
        - /dataset1/
            - /dataset1_converter.py
            - /TLiDB_dataset1/
                - /dataset1 files
        - /dataset2/
            - /dataset2_converter.py
            - /TLiDB_dataset2/
                - /dataset2 files
    - /other_internal_files
