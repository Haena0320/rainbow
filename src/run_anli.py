import os, sys
sys.path.append("/home/user15/workspace/rainbow")
import argparse
import glob
import logging
import numpy as np
from io import open
import torch
from torch.cuda import amp
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, Dataset

from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers import RobertaForMultipleChoice, BertForMultipleChoice

from utils.common import *
from nn_utils.help import *

class AnliDataset(Dataset):
    LABELS = ["1",'2']
    DATA_TYPE_TO_FILENAME = {
        "train":"alphanli-train-dev/train.jsonl",
        "train_label":"alphanli-train-dev/train-labels.lst",
        "dev":"alphanli-train-dev/dev.jsonl",
        "dev_label":"alphanli-train-dev/dev-labels.lst",
        "test":"alphanli-test/anli.jsonl"}

    @staticmethod
    def get_labels():
        return [0,1]

    def __init__(self, model_class, data_type, data_dir, tokenizer, do_lower_case, max_seq_length, **kwargs):
        self.model_class = model_class
        self.data_type = data_type
        self.data_dir = data_dir  # datasets/winogrande
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length

        self.data_path = os.path.join(self.data_dir, self.DATA_TYPE_TO_FILENAME[self.data_type])
        raw_examples = load_jsonl(self.data_path)
        if self.data_type != 'test':
            label_path = os.path.join(self.data_dir, self.DATA_TYPE_TO_FILENAME[self.data_type+"_label"])
            raw_labels = load_text(label_path)
            assert len(raw_examples) == len(raw_labels)
        self.example_list = []
        for i, line in enumerate(raw_examples):
            qid = line["story_id"]
            obs1 = line["obs1"]
            obs2 = line["obs2"]
            hyp1 = line["hyp1"]
            hyp2 = line["hyp2"]
            label = raw_labels[i] if self.data_type != "test" else "1"

            example = {
                "qid": qid,
                "obs1": obs1,
                "obs2": obs2,
                "hyp1": hyp1,
                "hyp2": hyp2,
                "label": self.LABELS.index(label)
            }
            self.example_list.append(example)

        self.cls_token, self.sep_token, self.pad_token = \
            self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token
        self.cls_id, self.sep_id, self.pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.cls_token, self.sep_token, self.pad_token])

    def get_all_qid(self):
        return [q["qid"] for q in self.example_list]

    def __getitem__(self, item):
        example = self.example_list[item]

        max_seq_length = self.max_seq_length
        obs1 = self.tokenizer.tokenize(example["obs1"])
        obs2 = self.tokenizer.tokenize(example["obs2"])
        hyp_tokens = [self.tokenizer.tokenize(example["hyp1"]), self.tokenizer.tokenize(example["hyp2"])]

        tokens = []
        token_ids = []
        segment_ids = []
        for choice_idx, hyp_token in enumerate(hyp_tokens):
            hyp = hyp_token + obs2
            _truncate_seq_pair(obs1, hyp, self.max_seq_length - 3)

            choice_tokens = []  # token id
            choice_segment_ids = []  # segment id
            choice_tokens.append(self.cls_token)
            choice_segment_ids.append(0)

            for obs1_token in obs1:
                choice_tokens.append(obs1_token)
                choice_segment_ids.append(0)
            choice_tokens.append(self.sep_token)
            choice_segment_ids.append(0)

            if self.model_class == "bert":
                for hyp_token in hyp:
                    choice_tokens.append(hyp_token)
                    choice_segment_ids.append(1)
                choice_segment_ids.append(1)

            else:
                choice_tokens.append(self.sep_token)
                choice_segment_ids.append(0)
                for hyp_token in hyp:
                    choice_tokens.append(hyp_token)
                    choice_segment_ids.append(0)
                choice_segment_ids.append(0)
            choice_tokens.append(self.sep_token)

            choice_token_ids = self.tokenizer.convert_tokens_to_ids(choice_tokens)

            tokens.append(choice_tokens)
            token_ids.append(choice_token_ids)
            segment_ids.append(choice_segment_ids)

        # padding
        cur_max_len = max(len(_e) for _e in tokens)
        choices_features = []

        for _idx_choice in range(len(tokens)):
            choice_tokens = tokens[_idx_choice]
            choice_token_ids = token_ids[_idx_choice]
            choice_segment_ids = segment_ids[_idx_choice]
            assert len(choice_tokens) <= max_seq_length, "{}/{}".format(len(choice_tokens), max_seq_length)
            assert len(choice_tokens) == len(choice_token_ids) == len(choice_segment_ids)

            padding_len = cur_max_len - len(choice_token_ids)
            padded_choice_token_ids = choice_token_ids + [self.pad_id] * padding_len
            padded_choice_token_mask = [1] * len(choice_token_ids) + [0] * padding_len
            padded_choice_segment_ids = choice_segment_ids + [0] * padding_len

            choices_features.append((choice_tokens, padded_choice_token_ids,
                                     padded_choice_token_mask, padded_choice_segment_ids))

        input_ids = torch.tensor([_e[1] for _e in choices_features], dtype=torch.long)
        mask_ids = torch.tensor([_e[2] for _e in choices_features], dtype=torch.long)
        segment_ids = torch.tensor([_e[3] for _e in choices_features], dtype=torch.long)
        label = torch.tensor(example["label"], dtype=torch.long)

        return input_ids, mask_ids, segment_ids, label

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t == 0:
                padding_value = self.pad_id
            else:
                padding_value = 0

            if _tensors[0].dim() >= 1:
                _tensors = [_t.t() for _t in _tensors]
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(
                        _tensors, batch_first=True, padding_value=padding_value).transpose(-1, -2),
                )
            else:
                return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list)

    def __len__(self):
        return len(self.example_list)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def evaluate(args, eval_dataset, model, tokenizer, global_step,
             is_saving_pred=False, file_prefix=""):
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader = setup_eval_step(
        args, eval_dataset, collate_fn=eval_dataset.data_collate_fn, )
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    _idx_ex = 0
    _eval_predict_data = []
    preds_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = [_t.to(args.device) for _t in batch]
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, token_type_ids=segment_ids,
                attention_mask=input_mask, labels=label_ids)
            tmp_eval_loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
        preds = np.argmax(logits, axis=-1)  # bn
        preds_list.append(preds)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy
              }

    qid_list = eval_dataset.get_all_qid()
    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results at {}*****".format(global_step))
        writer.write("***** Eval results at {}*****\n".format(global_step))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n")
    return eval_accuracy


def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """ Train the model """
    # learning setup
    train_dataloader = setup_training_step(
        args, train_dataset, collate_fn=train_dataset.data_collate_fn)
    model, optimizer, scheduler = setup_opt(args, model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)

    global_step = 0
    best_accu = 0.
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    scalar = amp.GradScaler()
    for _idx_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps))
        step_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with amp.autocast():
                outputs = model(
                    input_ids=input_ids, token_type_ids=segment_ids,
                    attention_mask=input_mask, labels=label_ids
                )
                loss = outputs[0]
                loss = update_wrt_loss(args, model, optimizer, loss, scalar)

                step_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    model_update_wrt_gradient(args, model, optimizer, scheduler, scalar)
                    global_step += 1
                    step_loss = 0.

                    if eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                        cur_accu = evaluate(args, eval_dataset, model, tokenizer, global_step=global_step)
                        if cur_accu > best_accu:
                            best_accu = cur_accu
                            save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

                    if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
        # evaluation each epoch or last epoch
        if (_idx_epoch == int(args.num_train_epochs) - 1) or (eval_dataset is not None and args.eval_steps <= 0):
            cur_accu = evaluate(args, eval_dataset, model, tokenizer, global_step=global_step)
            if cur_accu > best_accu:
                best_accu = cur_accu
                save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
        fp.write("{}{}".format(best_accu, os.linesep))


def main():
    parser = argparse.ArgumentParser()
    # data related
    parser.add_argument("--dataset", default="anli", type=str,
                        help="[anli|cosmosqa|hellaswag|physicaliqa|socialiqa|winogrande]")
    parser.add_argument("--model_class", default="roberta", type=str, help="[roberta|bert]")
    parser.add_argument("--data_dir", required=True, type=str, help="")
    parser.add_argument("--data_split", default="rand", type=str, help="The input data dir.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    define_hparams_training(parser)
    args = parser.parse_args()
    setup_prerequisite(args)
    print(args.device)
    if args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMultipleChoice
    elif args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        model_class = BertForMultipleChoice
    else:
        raise KeyError(args.model_class)

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    train_dataset = AnliDataset(args.model_class,"train", args.data_dir, tokenizer, args.do_lower_case, args.max_seq_length)
    dev_dataset = AnliDataset(args.model_class, "dev", args.data_dir, tokenizer, args.do_lower_case, args.max_seq_length)
    # test_dataset = AnliDataset("test", args.data_dir, tokenizer, args.do_lower_case, args.max_seq_length)

    if args.do_train:
        train(args, train_dataset, model, tokenizer, dev_dataset)

    if args.do_train and args.do_eval:  # load the best model
        model = model_class.from_pretrained(args.output_dir, config=config)
        model.to(args.device)

    if args.do_eval:
        dev_accu = evaluate(args, dev_dataset, model, tokenizer, is_saving_pred=True, global_step=None,
                            file_prefix="dev_")
        # test_accu = evaluate(args, test_dataset, model, tokenizer, is_saving_pred=True, global_step=None, file_prefix="aux_")
        with open(os.path.join(args.output_dir, "predicted_results.txt"), "a") as fp:
            # fp.write("{},{}{}".format(dev_accu, test_accu, os.linesep))
            fp.write("****** hyperparams ****** {}".format(os.linesep))
            fp.write("adam_epsilon {}{}".format(args.adam_epsilon, os.linesep))
            fp.write("weight_decay {}{}".format(args.weight_decay, os.linesep))
            fp.write("max_seq_length {}{}".format(args.max_seq_length, os.linesep))
            fp.write("learning_rate {}{}".format(args.learning_rate, os.linesep))
            fp.write("max_steps {}{}".format(args.max_steps, os.linesep))
            fp.write("eval_steps {}{}".format(args.eval_steps, os.linesep))
            fp.write("train_batch_size {}{}".format(args.train_batch_size, os.linesep))
            fp.write("eval_batch_size {}{}".format(args.eval_batch_size, os.linesep))
            fp.write("gradient_accumulation_steps {}{}".format(args.gradient_accumulation_steps, os.linesep))
            fp.write("seed {}{}".format(args.seed, os.linesep))
            fp.write("******* results *********{}".format(os.linesep))
            fp.write("{}{}".format(dev_accu, os.linesep))


if __name__ == '__main__':
    main()