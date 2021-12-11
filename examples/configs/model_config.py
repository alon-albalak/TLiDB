t5_config = {
    "model": "t5-base",
    "optimizer": "Adafactor",
    "learning_rate": 3e-5,
    "fp16": False,
    "effective_batch_size":128
}
gpt2_config = {
    "model": "gpt2",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "effective_batch_size":128
}
bert_config = {
    "model": "bert-base-uncased",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "effective_batch_size":40
}