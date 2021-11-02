#!/usr/bin/env bash

wget -O train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
wget -O test.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
wget -O categories.json https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json

python3 convert_banking77.py

rm train.csv
rm test.csv
rm categories.json
