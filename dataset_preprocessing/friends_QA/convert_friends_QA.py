import os
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()

def format_data(original_data, formatted_data, partition):
    """Updates formatted_data inplace with the data from original_data"""
    for scene in original_data['data']:
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
                "dialogue_id":f"{qa['id']}_{partition}",
                "dialogue_metadata":{
                    "question_answering":None,
                    "original_data_partition":partition
                },
                "dialogue":dialogue,
                "question_answering":{
                    "query":qa['question'],
                    "answers":qa['answers']
                }
            }
            formatted_data['data'].append(formatted_datum)
    return


TLiDB_path="TLiDB_friends_QA"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "friends_QA",
        "tasks": [
            "question_answering"
        ],
        "task_metadata": {"question_answering": {"metrics": ["exact_match", "token_f1"]}}
    },
    "data": []
}


data_partitions = [["trn","train"],["dev","dev"],["tst","test"]]
for p in data_partitions:
    data_path=f"friendsqa_{p[0]}.json"
    original_data = json.load(open(data_path,"r"))
    format_data(original_data, formatted_data, p[1])

with open(os.path.join(TLiDB_path,f"TLiDB_friends_QA.json"), "w") as f:
    json.dump(formatted_data, f, indent=2)
