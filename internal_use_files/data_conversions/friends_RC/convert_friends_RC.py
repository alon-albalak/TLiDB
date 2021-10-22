import os
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()


def format_data(data):
    formatted_data = {
        "metadata":
        {
            "dataset_name":"friends_RC",
            "tasks":[
                "reading_comprehension"
            ]
        },
        "data":[]
    }

    scenes = {}
    for query in data:
        if query['scene_id'] not in scenes:
            scenes[query['scene_id']] = 0
        
        formatted_datum = {
            "dialogue_id":f"{query['scene_id']}_q{scenes[query['scene_id']]:02d}",
            "dialogue_metadata":{
                "reading_comprehension":None
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
    return formatted_data

TLiDB_path="TLiDB_friends_RC"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

data_partitions = [["trn","train"],["dev","dev"],["tst","test"]]
for p in data_partitions:
    data_path=f"reading-comprehension-{p[0]}.json"
    data = json.load(open(data_path,"r"))
    formatted_data = format_data(data)
    with open(os.path.join(TLiDB_path,f"reading-comprehension-{p[1]}.json"), "w") as f:
        json.dump(formatted_data, f, indent=2)