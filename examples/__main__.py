from .argparser import parse_args
from .run_experiment import main as run_experiment

def main():
    config = parse_args()
    run_experiment(config)