# import torch
from algorithms import initialize_algorithm
from train import train,evaluate
import sys
import os
from utils import Logger, load_datasets_split, load_algorithm, log_config, \
        set_seed, log_dataset_info, get_savepath_dir, append_to_save_path_dir
import argparser

def main(config):

    config.save_path_dir = get_savepath_dir(config.train_datasets, config.train_tasks, config.seed, config.log_and_model_dir, config.model)

    # Initialize logs
    if os.path.exists(config.save_path_dir) and \
        (config.resume or ((config.do_finetune or config.do_eval) and (not config.do_train))):
        # if explicitly resuming, or running eval only then append to logger
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.save_path_dir):
        os.makedirs(config.save_path_dir)
    logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode)

    set_seed(config.seed)

    # load datasets for training
    datasets = {}
    if config.do_train:
        logger.write("TRAINING\n")
        datasets['train'] = load_datasets_split("train",config)
        datasets['dev'] = load_datasets_split("dev",config)

        # log configuration
        log_config(config,logger)
        # log dataset info
        log_dataset_info(datasets, logger)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        resume_success = False
        if resume:
            if os.path.exists(os.path.join(config.save_path_dir, 'last_model.pt')):
                prev_epoch, best_val_metric = load_algorithm(algorithm, os.path.join(config.save_path_dir, 'last_model.pt'),logger)
                epoch_offset = prev_epoch + 1
                logger.write(f"Resuming training from epoch {prev_epoch} with best validation metric {best_val_metric}\n")
                resume_success = True
            else:
                logger.write("No previous model found, starting from scratch\n")

        if not resume_success:
            epoch_offset=0
            best_val_metric = None

        train(algorithm, datasets, config, logger, epoch_offset, best_val_metric)

    if config.do_finetune:
        logger.write("FINETUNING\n")
        # if fine tuning, set fine-tune train, and fine-tune dev to the same tasks
        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        datasets['train'] = load_datasets_split("finetune_train",config)
        datasets['dev'] = load_datasets_split("finetune_dev",config)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        # always load best model
        model_path = os.path.join(config.save_path_dir, 'best_model.pt')
        is_best = True
        
        load_algorithm(algorithm, model_path, logger)
        epoch_offset = 0
        best_val_metric = None

        config.save_path_dir = append_to_save_path_dir(config.save_path_dir, config.finetune_datasets, config.finetune_tasks, config.seed)
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)

        # log configuration
        log_config(config,logger)
        # log dataset info
        log_dataset_info(datasets, logger)

        train(algorithm, datasets, config, logger, epoch_offset, best_val_metric)

    if config.do_eval:
        assert(config.do_finetune or config.eval_model_dir is not None), "Must specify --eval_model_dir"
        assert(not(config.eval_last and config.eval_best)), "cannot evaluate both last and best models"
        assert(config.eval_last or config.eval_best), "must evaluate at least one model"
        logger.write("EVALUATING\n")
        
        datasets['test'] = load_datasets_split("test",config)

        # log dataset info
        log_dataset_info(datasets, logger)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)  
        
        if config.do_finetune:
            config.eval_model_dir = config.save_path_dir
        if config.eval_last:
            eval_model_path = os.path.join(config.eval_model_dir, "last_model.pt")
            is_best = False
        else:
            eval_model_path = os.path.join(config.eval_model_dir, 'best_model.pt')
            is_best = True

        epoch, best_val_metric = load_algorithm(algorithm, eval_model_path,logger)
        evaluate(algorithm, datasets, config, logger, epoch, is_best)

    logger.close()

if __name__ == "__main__":
    config = argparser.parse_args()
    main(config)
