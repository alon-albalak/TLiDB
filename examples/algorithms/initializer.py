from algorithms.EncoderAlgorithm import EncoderAlgorithm
from algorithms.Seq2SeqAlgorithm import Seq2SeqAlgorithm

def initialize_algorithm(config, datasets):
    """Load an algorithm of type Algorithm
    Args:
        config (dict): configuration dictionary
        datasets (dict): dictionary of datasets
    Returns:    
        algorithm (Algorithm): an algorithm object
    """
    if config.model_type == "Encoder":
        algorithm = EncoderAlgorithm(config, datasets)
    elif config.model_type == "Seq2Seq":
        algorithm = Seq2SeqAlgorithm(config, datasets)
    else:
        raise ValueError(f"Invalid algorithm name: {config.model_type}")

    return algorithm