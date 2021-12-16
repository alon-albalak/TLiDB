GPU=$1
SOURCE_TASK=$2
MODEL=$3
GPU_BATCH_SIZE=$4

train_eval_source(){
    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        --model_config $MODEL \
        --gpu_batch_size $GPU_BATCH_SIZE \
        --do_train \
        --source_tasks $SOURCE_TASK --source_datasets DailyDialog \
        --num_epochs=10 \
        --do_eval --eval_best --target_tasks $SOURCE_TASK --target_datasets DailyDialog
}

finetune_eval_target(){
    TARGET_TASK=$1

    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        --model_config $MODEL \
        --gpu_batch_size $GPU_BATCH_SIZE \
        --source_tasks $SOURCE_TASK --source_datasets DailyDialog \
        --do_finetune \
        --target_tasks $TARGET_TASK --target_datasets DailyDialog \
        --num_epochs=10 \
        --do_eval --eval_best
}


tasks=(
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

train_eval_source

for task in ${tasks[@]}; do
    if [ $task != $SOURCE_TASK ]; then
        finetune_eval_target $task
    fi
done
