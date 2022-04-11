# Model configurations can be used to simplify calls to run_experiment
# Any configuration normally passed to run_experiment can be passed here
#   and will act the same as if passed on the command line


t5_config = {
    "model": "t5-base",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": False,
    "max_dialogue_length": 512 # at most 512 words
}
gpt2_config = {
    "model": "gpt2",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "max_dialogue_length": 512 # at most 512 words
}
bert_config = {
    "model": "bert-base-uncased",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "max_dialogue_length": 0 # no maximum legnth
}