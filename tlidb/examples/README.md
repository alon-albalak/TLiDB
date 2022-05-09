# Training Examples
This folder contains examples of how to use TLiDB. Including the following:
1. Model Architectures:
    - BERT (Encoder)
    - GPT-2 (Decoder)
    - T5 (EncoderDecoder)
2. Learning Algorithms:
    - Pre-train/Fine-tune
    - Multitask
    - Multitask/Fine-tune
3. Sample training scripts

## Sample Models
New models can easily be added by including a new model.py file in `/examples/models`. If the model is not an encoder-only, decoder-only, or encoder-decoder model, then you will need to implement an algorithm for your model in `/examples/algorithms`

Default models include: BERT, GPT-2, and T5

## Sample Learning Algorithms
We include 3 learning algorithms for transfer: Pre-train/fine-tune, Multitask, and Multitask/fine-tune.

## Sample Training Scripts
TLiDB has example scripts to be used for training and evaluating models in transfer learning settings: `run_experiment.py`. The following scripts should be run from the `/examples/` folder.

To simply train/evaluate a model on a single dataset/task:
```bash
MODEL=bert
DATASET=Friends
TASK=emory_emotion_recognition
python3 run_experiment.py --do_train --model_config $MODEL --source_tasks $TASK --source_datasets $DATASET --do_eval --eval_best --target_tasks $TASK --target_datasets $DATASET
```

To train a model on a source dataset/task and subsequently finetune on a target dataset/task:
```bash
MODEL=bert
SOURCE_DATASET=Friends
SOURCE_TASK=emory_emotion_recognition
TARGET_DATASET=Friends
TARGET_TASK=reading_comprehension
python3 run_experiment.py --do_train --model_config $MODEL --source_tasks $SOURCE_TASK --source_datasets $SOURCE_DATASET --do_finetune --do_eval --eval_best --target_tasks $TARGET_TASK --target_datasets $TARGET_DATASET
```

### Multitasking
In addition to the pre-train/fine-tune algorithm shown above, TLiDB also supports multitask training on the source and target dataset/task with a simple flag (`--multitask`), as in:
```bash
MODEL=bert
SOURCE_DATASET=Friends
SOURCE_TASK=emory_emotion_recognition
TARGET_DATASET=Friends
TARGET_TASK=reading_comprehension
python3 run_experiment.py --do_train --model_config $MODEL --source_tasks $SOURCE_TASK --source_datasets $SOURCE_DATASET --do_eval --eval_best --target_tasks $TARGET_TASK --target_datasets $TARGET_DATASET --multitask
```

Multitask training will validate only on the target task(s).
To Multitask and then fine-tune on the target dataset/task, simply include the `--do_finetune` flag.

### Large-scale training
TLiDB makes training on a single source and then fine-tuning on many datasets/tasks very simple. First, train the model on the source dataset/task:
```bash
MODEL=t5
SOURCE_DATASET=Friends
SOURCE_TASK=emory_emotion_recognition
python3 run_experiment.py --do_train --model_config $MODEL --source_tasks $SOURCE_TASK --source_datasets $SOURCE_DATASET
```
Then, fine-tune on many target datasets/tasks:
```bash
MODEL=t5
SOURCE_DATASET=Friends
SOURCE_TASK=emory_emotion_recognition
TARGET_TASKS=(
    'reading_comprehension'
    'character_identification'
    'question_answering'
    'personality_detection'
    )
TARGET_DATASET=Friends
for target_task in ${TARGET_TASKS[@]}; do
    python3 run_experiment.py --do_finetune --model_config $MODEL --source_tasks $SOURCE_TASK --source_datasets $SOURCE_DATASET --target_tasks $target_task --target_datasets $TARGET_DATASET
done
```

### Model Configurations
For simplicity, we include model configurations in `/configs/model_config.py` which contains arguments to be parsed as if they were passed in from the command line. To simplify your own experiments, we recommend putting as many arguments as possible in the `model_config.py` file for your model, to ensure that no parameters are accidentally skipped.

Sample models can be called using `--model_config <bert|gpt2|t5>`

To create your own model config, simply create a new dictionary entry in `model_config.py` with the name of your model. For example, to create a new model config for bert-large, you would create a dictionary entry like:
```python3
bert_large_config = {
    "model": "bert-large-uncased",
    ...
}
```
and then use `--model_config bert_large` in the call to `run_experiment.py`.

## Example Scripts
We provide the example scripts `run_full_data_task_transfer_single_source.sh` and `run_few_shot_task_transfer_single_source.sh` to demonstrate a simple script used to run many transfer experiments.
The scripts will first train a model on the designated source task, then fine-tune and evaluate on each of the target tasks. These scripts are only meant to be a starting point.
