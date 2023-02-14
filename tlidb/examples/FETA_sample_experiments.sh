GPU=$1

FRIENDS_DATASETS=(
    'emory_emotion_recognition'
    'MELD_emotion_recognition'
    'reading_comprehension'
    'character_identification'
    'question_answering'
    'personality_detection'
    'relation_extraction'
)

FRIENDS_SOURCE_DATASETS=(
    'personality_detection'
    'relation_extraction'
    'MELD_emotion_recognition'
    'reading_comprehension'
    'reading_comprehension'
    'character_identification'
    'emory_emotion_recognition'
)

DD_DATASETS=(
    'emotion_recognition'
    'dialogue_act_classification'
    'topic_classification'
    'causal_emotion_span_extraction'
    'causal_emotion_entailment'
    'dialogue_nli'
    'dialogue_reasoning_span_extraction'
    'dialogue_reasoning_multiple_choice_span_selection'
    'dialogue_reasoning_commonsense_relation_prediction'
    'adversarial_response_selection'
)

DD_SOURCE_DATASETS=(
    'dialogue_act_classification'
    'emotion_recognition'
    'dialogue_nli'
    'causal_emotion_entailment'
    'causal_emotion_span_extraction'
    'adversarial_response_selection'
    'dialogue_act_classification'
    'adversarial_response_selection'
    'dialogue_reasoning_multiple_choice_span_selection'
    'topic_classification'
)

train_eval_source(){
    SOURCE_TASK=$1
    DATASET=$2

    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        --log_and_model_dir "baselines" \
        --model_config bert \
        --seed 42 \
        --gpu_batch_size 20 \
        --do_train \
        --source_tasks $SOURCE_TASK --source_datasets $DATASET \
        --num_epochs 10 \
        --do_eval --eval_best --target_tasks $SOURCE_TASK --target_datasets $DATASET \
        --few_shot_percent 0.1 \
        --save_pred

    find ./baselines -name "best_model.pt" -delete
}

finetune_eval_target(){
    SOURCE_TASK=$1
    DATASET=$2
    TARGET_TASK=$3

    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        --log_and_model_dir "multitask_train_finetune" \
        --model_config bert \
        --seed 42 \
        --gpu_batch_size 20 \
        --do_train \
        --multitask \
        --source_tasks $SOURCE_TASK --source_datasets $DATASET \
        --do_finetune \
        --target_tasks $TARGET_TASK --target_datasets $DATASET \
        --num_epochs 10 \
        --do_eval --eval_best \
        --few_shot_percent 0.1 \
        --save_pred

    find ./multitask_train_finetune -name "best_model.pt" -delete
}

for dataset in ${FRIENDS_DATASETS[@]}; do
    echo "Training baseline on $dataset"
    train_eval_source $dataset "Friends"
done

for dataset in ${DD_DATASETS[@]}; do
    echo "Training baseline on $dataset"
    train_eval_source $dataset "DailyDialog"
done

for i in "${!FRIENDS_DATASETS[@]}"; do
    echo "Multi-tasking on ${FRIENDS_DATASETS[$i]} and ${FRIENDS_SOURCE_DATASETS[$i]}, followed by finetuning on ${FRIENDS_DATASETS[$i]}"
    finetune_eval_target ${FRIENDS_SOURCE_DATASETS[$i]} "Friends" ${FRIENDS_DATASETS[$i]}
done

for i in "${!DD_DATASETS[@]}"; do
    echo "Multi-tasking on ${DD_DATASETS[$i]} and ${DD_SOURCE_DATASETS[$i]}, followed by finetuning on ${DD_DATASETS[$i]}"
    finetune_eval_target ${DD_SOURCE_DATASETS[$i]} "DailyDialog" ${DD_DATASETS[$i]}
done