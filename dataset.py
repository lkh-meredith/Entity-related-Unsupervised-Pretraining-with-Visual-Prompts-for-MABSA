import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer,AutoTokenizer
from itertools import chain
import pickle
import os
import sys
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer
import string

class CocoIterableDatasetForT5InstructionLMDual(torch.utils.data.IterableDataset):
    def __init__(self, model_path, feature_path, prefix_text_len, constant_len, num):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        self.padding_token_id = self.tokenizer.pad_token_id
        self.prefix_text_len = prefix_text_len
        self.max_len = 60
        self.feature_path = feature_path
        self.num = num
        self.constant_len = constant_len
        self.all_image_embeddings = None
        self.all_image_caption = None

    def __len__(self):
        return self.num

    def padding_to_max_len(self, input_ids, max_len):
        padding_len = max_len - len(input_ids)
        padding_input_tokens = input_ids
        if padding_len > 0:
            padding_input_tokens = torch.cat((input_ids, torch.zeros(padding_len, dtype=torch.int64)))
        elif padding_len < 0:
            padding_input_tokens = input_ids[:max_len]

        return padding_input_tokens

    def shift_right_and_padding(self, input_ids, max_len):
        decoder_start_token_id = self.padding_token_id  # self.config.decoder_start_token_id 0
        pad_token_id = self.padding_token_id  # self.config.pad_token_id 0

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return self.padding_to_max_len(shifted_input_ids, max_len)

    def __iter__(self):
        file_list = os.listdir(self.feature_path)
        for file in file_list:
            if file.endswith('.pkl'):
                with open(os.path.join(self.feature_path, file), 'rb') as f:
                    feature = pickle.load(f)
                    self.all_image_embeddings = feature['image_embedding']
                    self.all_image_caption = feature['captions']
                    for index in range(len(self.all_image_embeddings)):
                        o = {}
                        image_feature = self.all_image_embeddings[index]
                        caption = self.all_image_caption[index]
                        caption_list = caption.split()
                        prefix_text_list = caption_list[0:self.prefix_text_len]
                        prefix_text = ' '.join(prefix_text_list)
                        prefix_text = "Caption of this image: " + prefix_text 
                        prefix_tokens = self.tokenizer(prefix_text,max_length=self.prefix_text_len+17,padding='max_length',truncation=True) 

                        surfix_text_list = caption_list[self.prefix_text_len:]
                        surfix_text = ' '.join(surfix_text_list)
                        surfix_tokens = self.tokenizer(surfix_text)

                        o["image_feature"]=image_feature
                        o["input_ids"] = torch.tensor(prefix_tokens["input_ids"])
                        o["attention_mask"] = torch.tensor(prefix_tokens['attention_mask'])
                        o["decoder_input_ids"] = self.shift_right_and_padding(torch.tensor(surfix_tokens['input_ids']),self.max_len) 
                        o["decoder_attention_mask"] = self.padding_to_max_len(torch.tensor(surfix_tokens['attention_mask']),self.max_len)

                        image_mask_label = torch.tensor([-100]*self.constant_len)
                        labels = torch.cat([image_mask_label,torch.tensor(surfix_tokens['input_ids'])])
                        o["labels"] = self.padding_to_max_len(labels,self.max_len+self.constant_len) 

                        yield o


def process_punctuation(sentence):
    punctuation = [',','.',':',';','-','_','@']
    for pun in punctuation:
        if sentence.find(pun) != -1:
            sentence=sentence.replace(pun,f" {pun}")
    return sentence

class CocoIterableDatasetForT5InstructionRCSDual(torch.utils.data.IterableDataset):
    def __init__(self, model_path, feature_path, mlm_probability,constant_len, num):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        self.feature_path = feature_path
        self.num = num
        self.max_len = 60
        self.decoder_max_len = 60
        self.pad_token_id = self.tokenizer.pad_token_id
        self.constant_len = constant_len
        self.mlm_probability = mlm_probability
        self.all_image_embeddings = None
        self.all_image_caption = None
        self.all_image_objects = None

    def __len__(self):
        return self.num

    def __iter__(self):
        file_list = os.listdir(self.feature_path)
        for file in file_list:
            if file.endswith('.pkl'):
                with open(os.path.join(self.feature_path, file), 'rb') as f:
                    feature = pickle.load(f)
                    self.all_image_embeddings = feature['image_embedding']
                    self.all_image_captions = feature['captions']
                    self.all_image_objects = feature['objects:']
                    for index in range(len(self.all_image_embeddings)):
                        o = {}
                        image_feature = self.all_image_embeddings[index]
                        caption = self.all_image_captions[index]
                        Instruction = "Caption of this image: "
                        Instruction_caption = Instruction + caption 
                        encode_process = process_punctuation(Instruction_caption) 
                        encoded = self.tokenizer(encode_process, max_length=self.max_len,
                                                padding='max_length', truncation=True, return_tensors='pt')  

                        caption_list = encode_process.split()
                        encoded_split_word_ids = encoded.word_ids()
                        objects = self.all_image_objects[index]
                     
                        if len(objects)>1:
                            probability_objects = torch.full((1, len(objects)), self.mlm_probability)
                            masked_objects_indices = torch.bernoulli(probability_objects).bool()  
                            choice_objects_list = []
                            for i, item in enumerate(masked_objects_indices.squeeze(0)):
                                if item:
                                    choice_objects_list.append(objects[i])
                        else:
                            choice_objects_list = objects

                        encode_process_input = encode_process
                        sentinel_token_dict = {}
                        for i, item in enumerate(choice_objects_list):
                            sentinel_token = f"<extra_id_{i}>"
                            sentinel_token_dict[sentinel_token] = self.tokenizer.convert_tokens_to_ids(sentinel_token)
                            encode_process_input = encode_process_input.replace(item, sentinel_token)
                        encode_input = self.tokenizer(encode_process_input, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

                        i = 0
                        objects_pos_list=[]
                        for obj in choice_objects_list:
                            obj_list = obj.split()
                            obj_len = len(obj_list)
                            index_i = 0
                            while i < len(caption_list) and index_i<obj_len:
                                if caption_list[i] == obj_list[index_i]:
                                    cur_index = index_i
                                    cur_i = i
                                    flag_cur = False
                                    cur_i_len = cur_i+obj_len
                                    cur_index_len = cur_index+obj_len
                                    while cur_i<cur_i_len and cur_index<cur_index_len:
                                        if caption_list[cur_i]!=obj_list[cur_index]:
                                            flag_cur = True
                                            break
                                        cur_i+=1
                                        cur_index+=1
                                    if not flag_cur:
                                        start_pos = i
                                        end_pos = i+obj_len-1
                                        objects_pos_list.append([start_pos, end_pos])
                                        i = end_pos+1
                                        break
                                    else:
                                        i += 1
                                    if obj_len <= 1:
                                        end_pos = start_pos
                                        objects_pos_list.append([start_pos, end_pos])
                                        i += 1
                                        break
                                else:
                                    i += 1

                        labels = [-100] * self.decoder_max_len
                        decoder_attention_mask = [0] * self.decoder_max_len
                        index_j = 0
                        encode_input_ids = encoded["input_ids"].squeeze(0)
                        for j, pos_pair in enumerate(objects_pos_list):
                            pos_start = pos_pair[0]
                            pos_end = pos_pair[1]
                            sentinel_token = f"<extra_id_{j}>"
                            labels[index_j]=sentinel_token_dict[sentinel_token]
                            decoder_attention_mask[index_j] = 1
                            index_j+=1
                            if pos_start == pos_end:
                               for index_i, item in enumerate(encoded_split_word_ids):
                                   if item == pos_start:
                                       labels[index_j]=int(encode_input_ids[index_i])
                                       decoder_attention_mask[index_j] = 1
                                       index_j += 1
                                   if item is None:
                                       break
                            else:
                                for index_i, item in enumerate(encoded_split_word_ids):
                                    if item == pos_start or item == pos_end:
                                        labels[index_j] = int(encode_input_ids[index_i])
                                        decoder_attention_mask[index_j] = 1
                                        index_j += 1
                                    if item is None:
                                        break
                       
                        labels = torch.tensor(labels)
                        decoder_for_input_ids = labels
                        decoder_for_input_ids.masked_fill_(decoder_for_input_ids == -100, self.pad_token_id)
                        labels[index_j] = self.tokenizer.eos_token_id

                        o["image_feature"] = image_feature
                        o["input_ids"] = encode_input["input_ids"].squeeze(0)
                        o["attention_mask"] = encode_input['attention_mask'].squeeze(0)
                        o["decoder_input_ids"] = decoder_for_input_ids 
                        o["decoder_attention_mask"] = torch.tensor(decoder_attention_mask)
                        image_mask_label = torch.tensor([-100] * self.constant_len)
                        shifted_labels = torch.zeros(labels.shape, dtype=image_mask_label.dtype).fill_(-100)
                        shifted_labels[0:-1] = labels[1:].clone()
                        o["labels"] = torch.cat([image_mask_label, shifted_labels])
                        yield o


class ABSAIterableDatasetForT5Instruction(torch.utils.data.IterableDataset):
    def __init__(self,model_path, feature_path, num, max_len, is_pair=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.feature_path = feature_path
        self.max_len = max_len
        self.num = num
        self.is_pair = is_pair

    def __len__(self):
        return self.num

    def __iter__(self):
        file_list = os.listdir(self.feature_path)
        for file in file_list:
            if file.endswith('.pkl'):
                with open(os.path.join(self.feature_path, file), 'rb') as f:
                    feature = pickle.load(f)
                    self.sentence_word = feature["sentences"]
                    self.image_features = feature["image_features"]
                    self.label_ids = feature["labels"]
                    self.pairs = feature["pairs"]
                    for index in range(len(self.sentence_word)):
                        o = {}
                        label_ids = self.label_ids[index]
                        sentence = self.sentence_word[index]
                        sentence = ' '.join(sentence)
                        sentence_with_instruction = "According to this image, extract the aspect term and predict its corresponding sentiment in the text : " + sentence
                        encoder_tokenized_input = self.tokenizer(sentence_with_instruction, truncation=True,
                                                                 max_length=self.max_len+20,
                                                                 padding="max_length")

                        decoder_tokenized_input = self.tokenizer(sentence, truncation=True,
                                                                 max_length=self.max_len,
                                                                 padding="max_length")
                        word_ids = decoder_tokenized_input.word_ids()
                        # word_ids[0] = None
                        input_word_ids = [i if i != None else -1 for i in word_ids]
                        split_word_ids = []  # split fine-grained label further

                        pre_word_idx = None
                        for word_index in word_ids:
                            if word_index is None or word_index >= len(label_ids):  # or word_index == 0
                                split_word_ids.append(-100)  # ignore
                            else:
                                if pre_word_idx != word_index:
                                    split_word_ids.append(label_ids[word_index])
                                else:
                                    split_word_ids.append(-100)
                            pre_word_idx = word_index

                        o["input_ids"] = torch.tensor(encoder_tokenized_input["input_ids"])
                        o["attention_mask"] = torch.tensor(encoder_tokenized_input["attention_mask"])
                        o["decoder_input_ids"] = torch.tensor(decoder_tokenized_input['input_ids'])
                        o["decoder_attention_mask"] = torch.tensor(decoder_tokenized_input['attention_mask'])
                        o['split_label'] = torch.tensor(split_word_ids)
                        o['image_feature'] = self.image_features[index]

                        if self.is_pair:
                            o['pair'] = str(self.pairs[index])
                            o["input_word_ids"] = torch.tensor(input_word_ids)
                        yield o


class MABSAIterableDatasetForT5InstructionDual(torch.utils.data.IterableDataset):
    def __init__(self,model_path, feature_path, num, max_len, constant_len, is_pair=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.constant_len = constant_len
        self.feature_path = feature_path
        self.max_len = max_len
        self.num = num
        self.is_pair = is_pair

    def __len__(self):
        return self.num

    def __iter__(self):
        file_list = os.listdir(self.feature_path)
        for file in file_list:
            if file.endswith('.pkl'):
                with open(os.path.join(self.feature_path, file), 'rb') as f:
                    feature = pickle.load(f)
                    self.sentence_word = feature["sentences"]
                    self.image_features = feature["image_features"]
                    self.label_ids = feature["labels"]
                    self.pairs = feature["pairs"]
                    for index in range(len(self.sentence_word)):
                        o = {}
                        label_ids = self.label_ids[index]
                        sentence = self.sentence_word[index]
                        sentence = ' '.join(sentence)
                        sentence_with_instruction = "According to this image, extract the aspect term and predict its corresponding sentiment in the text : " + sentence
                      
                        encoder_tokenized_input = self.tokenizer(sentence_with_instruction, truncation=True,
                                                                 max_length=self.max_len+20,
                                                                 padding="max_length")

                        decoder_tokenized_input = self.tokenizer(sentence, truncation=True,
                                                                 max_length=self.max_len,
                                                                 padding="max_length")
                        word_ids = decoder_tokenized_input.word_ids()
                        
                        input_word_ids = [i if i != None else -1 for i in word_ids]
                        split_word_ids = []  # split fine-grained label further

                        pre_word_idx = None
                        for word_index in word_ids:
                            if word_index is None or word_index >= len(label_ids):  
                                split_word_ids.append(-100)  # ignore
                            else:
                                if pre_word_idx != word_index:
                                    split_word_ids.append(label_ids[word_index])
                                else:
                                    split_word_ids.append(-100)
                            pre_word_idx = word_index

                        o["input_ids"] = torch.tensor(encoder_tokenized_input["input_ids"])
                        o["attention_mask"] = torch.tensor(encoder_tokenized_input["attention_mask"])
                        o["decoder_input_ids"] = torch.tensor(decoder_tokenized_input['input_ids'])
                        o["decoder_attention_mask"] = torch.tensor(decoder_tokenized_input['attention_mask'])

                        image_mask_label = torch.tensor([-100] * self.constant_len)
                        split_label = torch.cat([image_mask_label, torch.tensor(split_word_ids)])
                        o['split_label'] = split_label
                        o['image_feature'] = self.image_features[index]

                        if self.is_pair:
                            o['pair'] = str(self.pairs[index])
                            o["input_word_ids"] = torch.tensor(input_word_ids)
                        yield o



