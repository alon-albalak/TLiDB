t5_config = {
    "model": "t5-base",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": False,
    "effective_batch_size":60,
}
gpt2_config = {
    "model": "gpt2",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "effective_batch_size":60,
}
bert_config = {
    "model": "bert-base-uncased",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "effective_batch_size":60,
}