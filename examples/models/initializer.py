def initialize_model(config, datasets):
    """
    Initialize models according to the configuration
    """
    if "bert" in config.model:
        model = initialize_bert_based_model(config, datasets)
    elif "gpt" in config.model:
        model = initialize_gpt2_based_model(config)
    elif "t5" in config.model:
        model = initialize_t5_based_model(config)
    return model


def initialize_bert_based_model(config, datasets):
    """
    Initialize BERT based model
    """
    from models.bert import Bert
    model = Bert(config, datasets)
    return model

def initialize_gpt2_based_model(config):
    """
    Initialize GPT2 based model
    """
    from models.gpt2 import GPT2
    model = GPT2(config)
    return model

def initialize_t5_based_model(config):
    """
    Initialize t5 based model
    """
    from models.t5 import T5
    model = T5(config)
    return model