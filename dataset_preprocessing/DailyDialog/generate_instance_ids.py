import json

TASK_TYPE_MAP={
    "emotion_recognition": "utt_level_classification",
    "dialogue_act_classification": "utt_level_classification",
    "topic_classification": "dial_level_classification",
    "causal_emotion_span_extraction": "span_extraction",
    "causal_emotion_entailment": "causal_emotion_entailment",
    "dialogue_nli": "dialogue_nli",
    "dialogue_reasoning_span_extraction": "dialogue_reasoning_span_extraction",
    "dialogue_reasoning_multiple_choice_span_selection": "multiple_choice",
    "dialogue_reasoning_commonsense_relation_prediction": "relation_extraction",
    "adversarial_response_selection": "adversarial_response_selection"
}

def generate_instance_ids(dataset):
    for datum in dataset['data']:
        for task in datum['dialogue_metadata']:
            if task == "original_data_partition":
                continue
            task_type = TASK_TYPE_MAP[task]
            if task_type == "utt_level_classification":
                for turn in datum['dialogue']:
                    if task in turn:
                        instance_id = f"{datum['dialogue_id']}_t{turn['turn_id']}"
                        turn[task] = {"label": turn[task], "instance_id": instance_id}
            elif task_type == "dial_level_classification":
                instance_id = datum['dialogue_id']
                datum[task] = {"label": datum[task], "instance_id": instance_id}
            elif task_type == "span_extraction":
                for qas in datum[task]:
                    for qa in qas['qas']:
                        qa['instance_id'] = qa['id']
                        del qa['id']
            elif task_type == "causal_emotion_entailment":
                for i, sample in enumerate(datum[task]):
                    instance_id = f"{datum['dialogue_id']}_cee{i}"
                    sample['instance_id'] = instance_id
            elif task_type == "dialogue_nli":
                for i, sample in enumerate(datum[task]):
                    instance_id = f"{datum['dialogue_id']}_dnli{i}"
                    sample['instance_id'] = instance_id
            elif task_type == "dialogue_reasoning_span_extraction":
                for i, qas in enumerate(datum[task]):
                    for j, qa in enumerate(qas['qas']):
                        instance_id = f"{datum['dialogue_id']}_context{i}_qa{j}"
                        qa['instance_id'] = instance_id
            elif task_type == "multiple_choice":
                for i, q in enumerate(datum[task]['mcqs']):
                    instance_id = f"{datum['dialogue_id']}_mcq{i}"
                    q['instance_id'] = instance_id
            elif task_type == "relation_extraction":
                for i, sample in enumerate(datum[task]):
                    instance_id = f"{datum['dialogue_id']}_re{i}"
                    sample['instance_id'] = instance_id
            elif task_type == "adversarial_response_selection":
                for i, sample in enumerate(datum[task]):
                    for j, triple in enumerate(sample['samples']):
                        instance_id = f"{datum['dialogue_id']}_context{i}_sample{j}"
                        triple['instance_id'] = instance_id
            else:
                raise ValueError(f"Unknown task type: {task_type}")


TLiDB_path="TLiDB_DailyDialog/TLiDB_DailyDialog.json"

# Load original DailyDialog data
dailydialog_data = json.load(open(TLiDB_path, "r"))

generate_instance_ids(dailydialog_data)

with open(TLiDB_path, "w") as f:
    json.dump(dailydialog_data, f, indent=2)