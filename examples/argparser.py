import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_and_model_dir", type=str, default="./logs_and_models")

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
    parser.add_argument("--save_best", type=bool, default=True)
    parser.add_argument("--save_last", type=bool, default=True)

    # evaluation args
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_best", action="store_true")
    parser.add_argument("--eval_last", action="store_true")
    

    # TODO: Set up supported tasks/datasets so these are choices from a specified list
    #       Convert to source/target tasks/datasets
    #           this requires train/eval for source, and train/eval/test for target
    parser.add_argument("--train_tasks",type=str,nargs='+')
    parser.add_argument("--train_datasets", type=str, nargs='+')
    parser.add_argument("--test_tasks", type=str, nargs='+')
    parser.add_argument("--test_datasets", type=str, nargs='+')
    # loss args
    parser.add_argument("--loss_functions", type=str, nargs='+')

    # algorithm args
    # parser.add_argument("--model_type", type=str, default="Encoder")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # misc. args
    parser.add_argument("--progress_bar", type=bool, default=True)
    parser.add_argument("--frac", type=float, default=1.0,
        help="Convenience parameter that scales down dataset size to specified fraction, for debugging purposes")
    parser.add_argument("--debug", action="store_true", help="Setup training for debugging. For example, no logging")
    parser.add_argument("--save_pred",action="store_true",help="Save predictions for each task")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()

    if "bert" in args.model:
        setattr(args, "output_type", "categorical")
        setattr(args, "model_type", "Encoder")
    else:
        setattr(args, "output_type", "token")
        setattr(args, "model_type", "Seq2Seq")

    if not args.cpu_only:
        setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    return args
