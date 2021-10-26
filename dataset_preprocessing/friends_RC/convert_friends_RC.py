import os
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()


def format_data(original_data, formatted_data, partition, scenes):
    """Updates formatted_data inplace with the data from original_data"""
    
    for query in original_data:
        if query['scene_id'] not in scenes:
            scenes[query['scene_id']] = 0
        
        formatted_datum = {
            "dialogue_id":f"{query['scene_id']}_q{scenes[query['scene_id']]:02d}",
            "dialogue_metadata":{
                "reading_comprehension":None,
                "original_data_partition":partition
            },
            "dialogue":[],
            "reading_comprehension":{
                "query":detokenizer.detokenize(query['query'].split()),
                "answer":query['answer']
            }
        }
        turn_id=0
        for ut in query['utterances']:
            formatted_turn={
                "turn_id":turn_id,
                "speakers":[ut['speakers'].split(" ")],
                "utterance":detokenizer.detokenize(ut['tokens'].split())
            }
            turn_id += 1
            formatted_datum['dialogue'].append(formatted_turn)
        formatted_data['data'].append(formatted_datum)
        scenes[query['scene_id']] += 1
    return

TLiDB_path="TLiDB_friends_RC"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "friends_RC",
        "tasks": [
            "reading_comprehension"
        ],
        "task_metadata": {
            "reading_comprehension": {"metrics": ["accuracy"]}
        }
    },
    "data": []
}
scenes = {}

data_partitions = [["trn","train"],["dev","dev"],["tst","test"]]
for p in data_partitions:
    data_path=f"reading-comprehension-{p[0]}.json"
    original_data = json.load(open(data_path,"r"))
    format_data(original_data, formatted_data, p[1],scenes)


with open(os.path.join(TLiDB_path,"TLiDB_friends_RC.json"), "w") as f:
    json.dump(formatted_data, f, indent=2)
