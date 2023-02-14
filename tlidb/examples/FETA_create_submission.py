import os

FRIENDS_DATASETS=[
    'emory_emotion_recognition',
    'MELD_emotion_recognition',
    'reading_comprehension',
    'character_identification',
    'question_answering',
    'personality_detection',
    'relation_extraction'
]

FRIENDS_SOURCE_DATASETS=[
    'personality_detection',
    'relation_extraction',
    'MELD_emotion_recognition',
    'reading_comprehension',
    'reading_comprehension',
    'character_identification',
    'emory_emotion_recognition',
]

DD_DATASETS=[
    'emotion_recognition',
    'dialogue_act_classification',
    'topic_classification',
    'causal_emotion_span_extraction',
    'causal_emotion_entailment',
    'dialogue_nli',
    'dialogue_reasoning_span_extraction',
    'dialogue_reasoning_multiple_choice_span_selection',
    'dialogue_reasoning_commonsense_relation_prediction',
    'adversarial_response_selection'
]

DD_SOURCE_DATASETS=[
    'dialogue_act_classification',
    'emotion_recognition',
    'dialogue_nli',
    'causal_emotion_entailment',
    'causal_emotion_span_extraction',
    'adversarial_response_selection',
    'dialogue_act_classification',
    'adversarial_response_selection',
    'dialogue_reasoning_multiple_choice_span_selection',
    'topic_classification'
]

baseline_path = "baselines/PRETRAINED_0.1_FEWSHOT_{}.{}_seed.42/bert-base-uncased/predictions.csv"
multitask_path = "multitask_train_finetune/MULTITASK_0.1_FEWSHOT_{}.{}_{}.{}_seed.42/bert-base-uncased/FINETUNED_0.1_FEWSHOT_{}.{}_seed.42/predictions.csv"

# Move Friends predictions into a single directory
sub_dir = "friends_submission"
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)

for target, source in zip(FRIENDS_DATASETS, FRIENDS_SOURCE_DATASETS):
    print(target, source)
    
    # Create a directory for each target dataset
    if not os.path.exists(os.path.join(sub_dir, target)):
        os.mkdir(os.path.join(sub_dir, target))
    # move baseline predictions
    baseline_score_file = baseline_path.format("Friends", target)
    os.system(f"cp {baseline_score_file} {os.path.join(sub_dir, target, 'baseline_predictions.csv')}")
    # move transfer predictions
    multitask_score_file = multitask_path.format("Friends", source, "Friends", target, "Friends", target)
    os.system(f"cp {multitask_score_file} {os.path.join(sub_dir, target, 'predictions.csv')}")
    # zip the contents of the submission directory
    os.system(f"cd {sub_dir} && zip -r submission.zip *")


# Move DailyDialog predictions into a single directory
sub_dir = "dd_submission"
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)
# Create a directory for each target dataset
for target, source in zip(DD_DATASETS, DD_SOURCE_DATASETS):
    print(target, source)
    
    if not os.path.exists(os.path.join(sub_dir, target)):
        os.mkdir(os.path.join(sub_dir, target))
    # move baseline predictions
    baseline_score_file = baseline_path.format("DailyDialog", target)
    os.system(f"cp {baseline_score_file} {os.path.join(sub_dir, target, 'baseline_predictions.csv')}")
    # move transfer predictions
    multitask_score_file = multitask_path.format("DailyDialog", source, "DailyDialog", target, "DailyDialog", target)
    os.system(f"cp {multitask_score_file} {os.path.join(sub_dir, target, 'predictions.csv')}")
    # zip the contents of the submission directory
    os.system(f"cd {sub_dir} && zip -r submission.zip *")