from typing import Callable, Dict
import torch
import numpy as np

def cal_f1(p_pred_labels,inputs_word_ids,p_pairs):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = inputs_word_ids[i].tolist()
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] == -1:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp > 1:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = int(pp - 2)
                    flag = True
                elif pp == 1:
                    if flag:
                        end_pos = word_ids[j]
                else:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
        true_pair = set(eval(p_pairs[i]))
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        correct_num += len(true_pair & pred_pair)
    return correct_num, gold_num, predict_num
  


