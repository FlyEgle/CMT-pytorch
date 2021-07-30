# -*- coding:utf-8 -*-
"""
Calculate the video accuracy
"""
import os
import json
from apex import parallel
import numpy as np
from scipy.special import softmax
from tqdm import tqdm 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', type=str, default="/data/jiangmingchao/data/AICutDataset/imagenet/r50_acc_result/")

def parse_file(data_file):
    lines = open(data_file).readlines()
    lines_json = [json.loads(x.strip()) for x in lines]
    total_length = len(lines_json)
    correct = 0

    for line in lines_json:
        pred = np.argmax(softmax(np.array(line["pred_logits"])))
        label  = line["real_label"]
        if pred == label:
            correct += 1
    return correct, total_length

# get the max value index
def argmax(data_list: list, num: int):
    data_dict = {x:data_list[x] for x in range(len(data_list))}
    sorted_data_dict = sorted(data_dict.items(), key=lambda k: (k[1], k[0]), reverse=True)
    argmax_data_dict = sorted_data_dict[:num]
    argmax_index = [x[0] for x in argmax_data_dict]
    return argmax_index


def acc_top_n(data_file, n=5):
    lines = open(data_file).readlines()
    lines_json = [json.loads(x.strip()) for x in lines]
    total_length = len(lines_json)
    correct = 0

    for line in tqdm(lines_json):
        pred_list = softmax(np.array(line["pred_logits"])).tolist()
        # print(pred_list)
        arg_index = argmax(pred_list, n)
        # print(arg_index)
        label  = line["real_label"]
        if label in arg_index:
            correct += 1
    return correct, total_length


# logits_file = "/data/jiangmingchao/data/AICutDataset/logits2"

args = parser.parse_args()

logits_file = args.logits_file

total_correct, total_num = 0, 0
if os.path.isdir(logits_file):
    for file in os.listdir(logits_file):
        file_path = os.path.join(logits_file, file)
        # correct, num = parse_file(file_path)
        correct, num = acc_top_n(file_path, n=1)
        total_correct += correct
        total_num += num

    print(f"Accuracy is {total_correct/total_num}")

elif os.path.isfile(logits_file):
    correct, num = parse_file(logits_file)
    total_correct += correct
    total_num += num
    print(f"Accuracy is {total_correct/total_num}")
