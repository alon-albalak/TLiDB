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
    if config.algorithm == "Encoder":
        algorithm = EncoderAlgorithm(config, datasets)
    elif config.algorithm == "Seq2Seq":
        algorithm = Seq2SeqAlgorithm(config, datasets)
    else:
        raise ValueError(f"Invalid algorithm name: {config.algorithm}")

    return algorithm