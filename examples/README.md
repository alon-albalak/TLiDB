# Examples to run experiments on TLiDB

## Usages
```
python3 run_experiment.py --model_config t5 --gpu_batch_size 4 --do_train --source_tasks emotion_recognition --source_datasets DailyDialog
```

## TODO
- [] Write a README.md to run experiments
    - [] How to test response generation part?
- [] Checklist for response generation
    - [] What does ElementwiseLoss (TLiDB/metrics/loss, losses.py) mean?
    - [] `#FIXME` in `models/gpt2.py`, `models/t5.py`
    - [] `#TODO` in `../TLiDB/metrics/all_metrics.py`
