import TLiDB.utils.special_tokens as st

t5_config = {
    "model": "t5-base",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": False,
    "effective_batch_size":60,
    "special_tokens":["speaker1:", "speaker2:", st.context_token, st.endcontext_token, st.answer_token],
}
gpt2_config = {
    "model": "gpt2",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "effective_batch_size":60,
    "special_tokens": ["speaker1:", "speaker2:", st.context_token, st.endcontext_token, st.answer_token],
}
bert_config = {
    "model": "bert-base-uncased",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "fp16": True,
    "effective_batch_size":60,
    "special_tokens": ["speaker1:", "speaker2:"]
}