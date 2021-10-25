import logging
import argparse
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)

def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Setting seed to {seed}")

def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # model args
    parser.add_argument("--model_name_or_path",type=str,required=True)
    # training args
    parser.add_argument("-e","--num_epochs",type=int,default=20)
    parser.add_argument("--effective_batch_size",type=int,default=10)
    parser.add_argument("--gpu_batch_size",type=int,default=10)
    parser.add_argument("-lr","--learning_rate",type=float,default=3e-5)
    parser.add_argument("--fp16",action="store_true")
    # data args
    parser.add_argument("--dataset_name",type=str,required=True)
    parser.add_argument("--task",type=str,required=True)
        
    args = parser.parse_args()

    if not args.cpu_only:
        setattr(args,"device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args,"device","cpu")

    return vars(args)
