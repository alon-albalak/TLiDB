import os
import pandas as pd
import json
import pdb


def format_data(original_data, formatted_data, partition):

    keys = {key: i for i, key in enumerate(original_data[0].split(','))}
    dialogue_ids = []
    for i, datum in enumerate(original_data[1:]):
        datum = datum.split(',')
        if datum[keys['conv_id']] not in dialogue_ids:
            dialogue_ids.append(datum[keys['conv_id']])
            formatted_datum = {
                "dialogue_id":datum[keys['conv_id']],
                "dialogue_metadata":{
                    "context":datum[keys['context']],
                    "prompt":datum[keys['prompt']],
                    "dialogue_generation":True,
                    "original_data_partition": partition
                },
                "dialogue":[]
            }
        formatted_turn = {
            "turn_id":datum[keys['utterance_idx']],
            "speakers":[datum[keys['speaker_idx']]],
            "utterance":datum[keys['utterance']],
        }
        formatted_datum['dialogue'].append(formatted_turn)
        if i+2 < len(original_data):
            if original_data[i+2].split(',')[keys['conv_id']] != datum[keys['conv_id']]:
                formatted_data['data'].append(dict(formatted_datum))
    formatted_data['data'].append(dict(formatted_datum))
    """
    dialogue_ids = []
    for i, datum in original_data.iterrows():
        if datum['conv_id'] not in dialogue_ids:
            dialogue_ids.append(datum['conv_id'])
            formatted_datum = {
                "dialogue_id":datum['conv_id'],
                "dialogue_metadata":{
                    "context":datum['context'],
                    "prompt":datum['prompt'],
                    "dialogue_generation":True,
                    "original_data_partition": partition
                },
                "dialogue":[]
            }
        formatted_turn = {
            "turn_id":datum['utterance_idx'],
            "speakers":[datum['speaker_idx']],
            "utterance":datum['utterance'],
        }
        formatted_datum['dialogue'].append(formatted_turn)
        if i+1 < len(original_data['conv_id']):
            if original_data['conv_id'][i+1] != original_data['conv_id'][i]:
                formatted_data['data'].append(formatted_datum)
    formatted_data['data'].append(formatted_datum)
    """
    return

TLiDB_path="TLiDB_empatheticdialog"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "empatheticdialog",
        "tasks": [
            "dialogue_response_generation",
        ],
        "task_metadata": {
            "dialogue_response_generation": {"metrics": ["BLEU","P@1,100"]}
        }
    },
    "data": []
}

original_data_path="original_empatheticdialog"
for root, dirs, files in os.walk(original_data_path):

    for file in files:
        if file.endswith(".csv"):
            partition=file.split('.')[0]
            f_path=os.path.join(root,file)
            #original_data = pd.read_csv(f_path, sep=",",error_bad_lines=False, warn_bad_lines=True)
            with open(f_path,'r') as fin:
                original_data = fin.readlines()
            format_data(original_data, formatted_data, partition)


with open(f"{TLiDB_path}/TLiDB_empatheticdialog.json", "w") as f:
    json.dump(formatted_data,f,indent=2)
