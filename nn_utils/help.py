from __future__ import absolute_import, division, print_function

import logging
import logging as logger
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

def define_hparams_training(parser):
    ## Required parameters
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    #                     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
    #                         ALL_MODELS))
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run eval on the test set and save predictions")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")


    # parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for training.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=None, type=str,
                        help='betas for Adam optimizer')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,  # ! change to propostion
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=-1,
                        help="Eval model every X updates steps. if X > 0")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")



def setup_logging(args):
    logger.basicConfig(format='%(asctime)s: %(message)s', level=logger.INFO, datefmt='%m/%d %I:%M:%S %p')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def setup_prerequisite(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # Create output directory if needed

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    setup_logging(args)
    set_seed(args)

def setup_training_step(args, train_dataset, **kwargs):
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    train_sampler = RandomSampler(train_dataset)
    batch_size = args.train_batch_size // args.gradient_accumulation_steps

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,batch_size=batch_size, **kwargs)

    # learning step
    if args.max_steps <= 0:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    else:
        t_total = args.max_steps
        args.num_train_epochs = math.ceil(
            1. * args.max_steps * args.gradient_accumulation_steps / len(train_dataloader))

    if args.warmup_steps < 0:
        args.warmup_steps = math.ceil(t_total * args.warmup_proportion)

    args.t_total = t_total
    return train_dataloader

def setup_eval_step(args, eval_dataset, **kwargs):
    # SequentialSampler : sample continuously
    # DistributedSampler : sample randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, **kwargs)
    return eval_dataloader



def setup_opt(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.adam_betas is not None:
        adam_betas = tuple(float(_f) for _f in args.adam_betas.split(","))
        assert len(adam_betas) == 2
    else:
        adam_betas = (0.9, 0.999)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,betas=adam_betas,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.t_total)
    return model, optimizer, scheduler

def update_wrt_loss(args, model, optimizer, loss, scalar):
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    scalar.scale(loss).backward()
    return loss

def model_update_wrt_gradient(args, model, optimizer, scheduler, scalar):
    scalar.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scalar.step(optimizer)
    scalar.update()
    scheduler.step()
    optimizer.zero_grad()

def save_model_with_default_name(output_dir, model, tokenizer, args_to_save=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Saving model checkpoint to %s", output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if args_to_save is not None:
        torch.save(args_to_save, os.path.join(output_dir, "training_args.bin"))




