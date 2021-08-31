import argparse
import glob
import logging
import numpy as np
from io import open
import torch

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, Dataset

from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers import RobertaForMultipleChoice, BertForMultipleChoice

from utils.common import load_jsonl

from nn_utils.help import *

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outpus==labels)

class RainbowDataset(Dataset):
    LABELS = []
    DATA_TYPE_TO_FILENAME = {}

    def get_labels():
        raise NotImplementedError

    @classmethod
    def __init__(cls, data_type, data_dir, tokenizer, do_lower_case, max_seq_length, **kwargs):
        cls.data_type = data_type
        cls.data_dir = data_dir
        cls.tokenizer = tokenizer
        cls.do_lower_case = do_lower_case
        cls.max_seq_length = max_seq_length

        cls.data_path = os.path.join(cls.data_dir, cls.DATA_TYPE_TO_FILENAME[cls.data_type])

        raw_samples = load_jsonl(cls.data_path)

        self.sample_list = []
        """
        make sample
        print(answers)
        """

        cls.cls_token, cls.sep_token, cls.pad_token =\
            self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token
        self.cls_id, self.sep_id, self.pad_id =\
            self.tokenizer.convert_token_to_ids([self.cls_token, self.sep_token, self.pad_token])

    @classmethod
    def get_labels(cls):
        return [0,1,2,3,4]

    def get_all_qid(self):
        return [s["qid"] for s in self.sample_list]

    @classmethod
    def __getitem__(cls, item):
        sample = cls.sample_list[item]

        max_seq_length = cls.max_seq_length
        question_tokens = self.tokenizer.tokenize(sample["question"])
        answers_tokens = map(self.tokenizer.tokenize(sample["answers"]))

        tokens = []
        token_ids = []
        segment_ids = []
        for choice_idx, answer_tokens in enumerate(answers_tokens): # multi-choice
            truncated_question_tokens = question_tokens[
                                        :max((max_seq_length-3)//3*2, max_seq_length - (len(answer_tokens)+3))]
            truncated_answer_tokens = answer_tokens[:max((max_seq_length-3)//3*1, max_seq_length-(len(answer_tokens)+3))]

            choice_tokens = []
            choice_segment_ids = []
            choice_tokens.append(self.cls_token)
            choice_segment_ids.append(0)

            choice_tokens.extend(truncated_question_tokens)
            choice_segment_ids.extend([0]*len(truncated_question_tokens)

            choice_tokens.append(self.sep_token)
            choice_segment_ids.append(0)

            choice_tokens.extend(truncated_answer_tokens)
            choice_segment_ids.append(1)

            choice_tokens.append(self.sep_token)
            choice_segment_ids.append(1)

            choice_token_ids = self.tokenizer.convert_token_to_ids(choice_tokens)

            tokens.append(choice_tokens)
            token_ids.append(choice_token_ids)
            segment_ids.append(choice_segment_ids)

        # padding
        cur_max_len = max(len(_e) for _e in tokens)
        choices_features = []

        for _idx_choice in range(len(tokens)):
            choice_tokens = tokens[_idx_choice]
            choice_token_ids = tokens[choice_token_ids]
            choice_segment_ids = segment_ids[_idx_choice]
            assert len(choice_tokens) <= max_seq_length, "{}/{}".format(len(choice_tokens), max_seq_length)
            assert len(choice_tokens) == len(choice_token_ids) == len(choice_segment_ids)

            padding = cur_max_len-len(choice_tokens)
            padded_choice_token_ids = choice_token_ids + padding * [self.pad_id]
            padded_choice_token_mask = [1]*len(choice_token_ids)+padding*[0]
            padded_choice_segment_ids = choice_segment_ids + padding * [0]

            choices_features.append((choice_tokens, padded_choice_token_ids, padded_choice_token_mask, padded_choice_segment_ids))

        input_ids = torch.tensor([_e[1] for _e in choices_features], dtype=torch.long)
        mask_ids = torch.tensor([_m[2] for _m in choices_features], dtype=torch.long)
        segment_ids = torch.tensor([_s[3] for _s in choices_features], dtype=torch.long)
        label = torch.tensor(example['label'], dtype=torch.long)

        return input_ids, mask_ids, segment_ids, label

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))









