import json
import numpy as np

def accuracy(out, labels):
    outputs = np.argmax(out, axis=-1)
    return np.sum(outputs==labels)

def load_json(file):
    with open(file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def load_jsonl(file):
    data_list = []
    with open(file, "r") as fp:
        for line in fp:
            data_list.append(json.loads(line))
    return data_list


def load_text(file):
    data_list = []
    with open(file, "r") as fp:
        for line in fp:
            data_list.append(fp.readline().strip())
    return data_list

