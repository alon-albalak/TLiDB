from algorithms.ERM import ERM

def initialize_algorithm(config, datasets):
    """Load an algorithm of type Algorithm
    Args:
        config (dict): configuration dictionary
        datasets (dict): dictionary of datasets
    Returns:    
        algorithm (Algorithm): an algorithm object
    """
    if config.algorithm == "ERM":
        algorithm = ERM(config, datasets)
    else:
        raise ValueError(f"Invalid algorithm name: {config.algorithm}")

    return algorithm