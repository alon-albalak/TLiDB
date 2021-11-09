#!/usr/bin/env bash

for domain in airline fastfood finance insurance media software
do
    echo $domain
    for split in train dev test
    do
        wget -O ${domain}_${split}.tsv https://raw.githubusercontent.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/master/data/paper_splits/splits_annotated_at_turn_level/$domain/$split.tsv
    done
done

python convert_multidogo.py

rm *.tsv