import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # model args
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    # training args
    parser.add_argument("-e", "--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--gpu_batch_size", type=int, default=5)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # data args
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    # loss args
    parser.add_argument("--loss_function", type=str, default="cross_entropy")
    # algorithm args
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # misc. args
    parser.add_argument("--progress_bar", type=bool, default=True)

    args = parser.parse_args()

    if "bert" in args.model:
        setattr(args, "output_type", "categorical")
    else:
        setattr(args, "output_type", "token")

    if not args.cpu_only:
        setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    return args
