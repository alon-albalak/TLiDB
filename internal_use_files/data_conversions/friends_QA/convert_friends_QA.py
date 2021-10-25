import os
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()

def format_data(data):
    formatted_data = {
        "metadata":
        {
            "dataset_name":"friends_QA",
            "tasks":[
                "question_answering"
            ],
            "task_metadata":{"question_answering":{"metrics":["exact_match","token_f1"]}}
        },
        "data":[]
    }

    for scene in data['data']:
        dialogue = []
        utterances = scene['paragraphs'][0]['utterances:']
        for utt in utterances:
            formatted_turn = {
                "turn_id": utt['uid'],
                "speakers":utt['speakers'],
                "utterance":detokenizer.detokenize(utt['utterance'].split())
            }
            dialogue.append(formatted_turn)
        qas = scene['paragraphs'][0]['qas']
        for qa in qas:
            formatted_datum = {
                "dialogue_id":qa['id'],
                "dialogue_metadata":{
                    "question_answering":None
                },
                "dialogue":dialogue,
                "question_answering":{
                    "query":qa['question'],
                    "answers":qa['answers']
                }
            }
            formatted_data['data'].append(formatted_datum)
    return formatted_data


TLiDB_path="TLiDB_friends_QA"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

data_partitions = [["trn","train"],["dev","dev"],["tst","test"]]
for p in data_partitions:
    data_path=f"friendsqa_{p[0]}.json"
    data = json.load(open(data_path,"r"))
    formatted_data = format_data(data)
    with open(os.path.join(TLiDB_path,f"friends_QA_{p[1]}.json"), "w") as f:
        json.dump(formatted_data, f, indent=2)