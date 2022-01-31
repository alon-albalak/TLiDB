# Examples to run experiments on TLiDB

## Installation
* response generation evaluation
    * `pip install git+https://github.com/Maluuba/nlg-eval.git@master`
    * `pip install bert-score`
    * `pip install sacrebleu`
    * `pip install nltk`

## Usages

* example 1
```
python3 run_experiment.py --model_config t5 --gpu_batch_size 4 --do_train --source_tasks emotion_recognition --source_datasets DailyDialog
```

* example 2
```
python3 run_experiment.py --model_config=gpt2 --source_tasks response_generation --source_datasets DailyDialog --do_train --debug --frac=0.1
```

## TODO
- [] Write a README.md to run experiments
    - [] How to test response generation part?
- [] Checklist for response generation
    - [] What does ElementwiseLoss (TLiDB/metrics/loss, losses.py) mean?
    - [] `#FIXME` in `models/gpt2.py`, `models/t5.py`
    - [] `#TODO` in `../TLiDB/metrics/all_metrics.py`
