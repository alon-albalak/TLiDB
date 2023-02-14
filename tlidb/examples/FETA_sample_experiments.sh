GPU=$1

FRIENDS_TASKS=(
    'emory_emotion_recognition'
    'MELD_emotion_recognition'
    'reading_comprehension'
    'character_identification'
    'question_answering'
    'personality_detection'
    'relation_extraction'
)

# Source tasks matching each target task, in order
FRIENDS_SOURCE_TASKS=(
    'personality_detection'
    'relation_extraction'
    'MELD_emotion_recognition'
    'reading_comprehension'
    'reading_comprehension'
    'character_identification'
    'emory_emotion_recognition'
)

DD_TASKS=(
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

# Source tasks matching each target task, in order
DD_SOURCE_TASKS=(
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

train_eval_target(){
    TARGET_TASK=$1
    DATASET=$2

    # Fine-tune and evaluate only on source task
    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        --log_and_model_dir "baselines" \
        --model_config bert \
        --seed 42 \
        --gpu_batch_size 20 \
        --do_train \
        --source_tasks $TARGET_TASK --source_datasets $DATASET \
        --num_epochs 10 \
        --do_eval --eval_best --target_tasks $TARGET_TASK --target_datasets $DATASET \
        --few_shot_percent 0.1 \
        --save_pred

    find ./baselines -name "best_model.pt" -delete
}

multitask_finetune_eval_target(){
    SOURCE_TASK=$1
    DATASET=$2
    TARGET_TASK=$3

    # Multi-task on source task and target task, then fine-tune and evaluate on target task
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

# iterate over all friends tasks to get baseline scores
for task in ${FRIENDS_TASKS[@]}; do
    echo "Training baseline on $task"
    train_eval_target $task "Friends"
done

# iterate over all dailydialog tasks to get baseline scores
for task in ${DD_TASKS[@]}; do
    echo "Training baseline on $task"
    train_eval_target $task "DailyDialog"
done

# iterate over all friends tasks, and respective source tasks, to get multitask transfer scores
for i in "${!FRIENDS_TASKS[@]}"; do
    echo "Multi-tasking on ${FRIENDS_TASKS[$i]} and ${FRIENDS_SOURCE_TASKS[$i]}, followed by finetuning on ${FRIENDS_TASKS[$i]}"
    multitask_finetune_eval_target ${FRIENDS_SOURCE_TASKS[$i]} "Friends" ${FRIENDS_TASKS[$i]}
done

# iterate over all dailydialog tasks, and respective source tasks, to get multitask transfer scores
for i in "${!DD_TASKS[@]}"; do
    echo "Multi-tasking on ${DD_TASKS[$i]} and ${DD_SOURCE_TASKS[$i]}, followed by finetuning on ${DD_TASKS[$i]}"
    multitask_finetune_eval_target ${DD_SOURCE_TASKS[$i]} "DailyDialog" ${DD_TASKS[$i]}
done