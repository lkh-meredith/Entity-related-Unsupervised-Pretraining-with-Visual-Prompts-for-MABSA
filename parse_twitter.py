import torch
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from transformers import ViTModel,AutoImageProcessor,CLIPVisionModel,BartTokenizer,CLIPModel
import collections


class TrainInputProcess:
    def __init__(self,
                 vision_model_path,
                 # text_model_path,
                 data_type=None,
                 output_dir=None,
                 data_text_dir=None,
                 data_image_dir=None):

        self.vision_model_path = vision_model_path
        self.output_dir = output_dir
        self.data_text_dir = data_text_dir
        self.data_image_dir = data_image_dir
        self.data_type = data_type
       
        self.data_splits = ['train','dev','test']
        self.text_surfix = '.txt'
        self.data_dict = dict()
        self.input = dict()

    def get_text_dataset_for_T5_instruction(self):
        for split in self.data_splits:
            data_file_name = split + self.text_surfix
            text_path = os.path.join(self.data_text_dir, data_file_name)
            sentence_d = collections.defaultdict(list)
            sentence_l = []
            image_l = []
            label_l = []
            pair_l = []
            with open(text_path, 'r', encoding="utf-8") as f:
                while True:
                    text = f.readline().rstrip('\n').split()
                    if text == []:
                        break
                    aspect = f.readline().rstrip('\n').split()
                    sentiment = f.readline().rstrip('\n')
                    image_path = f.readline().rstrip('\n')
                    start_pos = text.index("$T$")
                    end_pos = start_pos + len(aspect) - 1
                    text = text[:start_pos] + aspect + text[start_pos + 1:]
                    sentence_d[" ".join(text)].append((start_pos, end_pos, sentiment, image_path))


    def get_text_dataset(self, process_label=False):
        for split in self.data_splits:
            data_file_name = split + self.text_surfix
            text_path = os.path.join(self.data_text_dir, data_file_name)
            sentence_d = collections.defaultdict(list)
            sentence_l = []
            image_l = []
            label_l = []
            pair_l = []
            with open(text_path,'r',encoding="utf-8") as f:
                while True:
                    text = f.readline().rstrip('\n').split()
                    if text == []:
                        break
                    aspect = f.readline().rstrip('\n').split()
                    sentiment = f.readline().rstrip('\n')
                    image_path = f.readline().rstrip('\n')
                    start_pos = text.index("$T$")
                    end_pos = start_pos + len(aspect) - 1
                    text = text[:start_pos] + aspect + text[start_pos+1:]
                    sentence_d[" ".join(text)].append((start_pos, end_pos, sentiment, image_path))
                  
                for key ,value in sentence_d.items():
                    text = key.split()
                    sentence_l.append(text)
                    n_key =len(text)
                    s_label = [0] * n_key
                    s_pair = []
                    image_l.append(value[0][3])
                    for vv in value:
                        v_sentiment = int(vv[2]) + 1
                        s_label[vv[0]] = v_sentiment + 2
                        for i in range(vv[0] + 1, vv[1] + 1): 
                            if process_label:
                                s_label[i] = v_sentiment + 4
                            else:
                                s_label[i] = 1
                        s_pair.append((str(vv[0]) + "-" + str(vv[1]), v_sentiment))
                    label_l.append(s_label)
                    pair_l.append(s_pair)
                self.data_dict[split] = (sentence_l, image_l, label_l, pair_l)

    def process_data(self):
        print(f"Begin to process data")
        model = CLIPVisionModel.from_pretrained(self.vision_model_path)
        image_processor = AutoImageProcessor.from_pretrained(self.vision_model_path)

        for split in self.data_splits:
            sentence_l, image_l, label_l, pair_l = self.data_dict[split]
            image_feature_l=[]
            file_num = 0
            start_index = 0
            for i in tqdm(range(len(image_l))):
                image_file_path = os.path.join(self.data_image_dir,image_l[i])
                image = Image.open(image_file_path)#.convert('RGB')
                image_input = image_processor(images=image, return_tensors='pt')
                image_embed = model(**image_input).last_hidden_state
                image_feature_l.append(image_embed)

                if (i + 1) % 300 == 0:
                    with open(os.path.join(self.output_dir,self.data_type,split, f"clip-vit-b16_{file_num}.pkl"), 'wb') as f:
                        pickle.dump({"sentences":sentence_l[start_index:i+1],"image_features":image_feature_l,"labels":label_l[start_index:i+1],"pairs":pair_l[start_index:i+1]}, f)
                        file_num += 1
                        start_index = i+1
                        image_feature_l = []

            with open(os.path.join(self.output_dir, self.data_type, split, f"clip-vit-b16_{file_num}.pkl"), 'wb') as f:
                pickle.dump({"sentences": sentence_l[start_index:], "image_features": image_feature_l, "labels": label_l[start_index:], "pairs": pair_l[start_index:]}, f)

            print(f"Save process input of {split} data")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model_path", type=str, default='./models/clip-vit-base-patch16')
    parser.add_argument("--data_type", type=str, default='twitter2017')
    parser.add_argument("--output_dir",type=str, default='./data/twitter/process')
    args = parser.parse_args()

    data_path = './data/twitter/' + args.data_type
    image_path = './data/twitter/images/' + args.data_type +'_images'
    processor = TrainInputProcess(vision_model_path=args.vision_model_path,
                                  data_type = args.data_type,
                                  data_text_dir=data_path,
                                  data_image_dir=image_path,
                                  output_dir=args.output_dir
                                  )

    processor.get_text_dataset()
    processor.process_data()
