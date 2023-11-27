import torch
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from transformers import ViTModel,AutoImageProcessor,CLIPVisionModel

def main(vision_model_path, data_path):
    for data_type in ['val2014','train2014']:
        out_path_process = f"{data_path}/mini_{data_type}_process_clip_features/mini_{data_type}_clip_process"
        model = CLIPVisionModel.from_pretrained(vision_model_path)
        processor = AutoImageProcessor.from_pretrained(vision_model_path)
        with open(f'{data_path}/annotations/mini_captions_{data_type}.json', 'r') as f:
            data = json.load(f)
        print("%0d captions loaded from json " % len(data['annotations']))

        all_embeddings = []
        all_captions = []
        file_num=0
        if data_type == 'val2014': #one caption for val
            image_id_list = list()
            store_num=0
            for i in tqdm(range(len(data['annotations']))):
                d = data['annotations'][i]
                img_id = d['image_id']
                if img_id not in image_id_list:
                    image_id_list.append(img_id)
                    all_captions.append(d['caption'])
                    filename = f"{data_path}/mini_{data_type}/COCO_{data_type}_{int(img_id):012d}.jpg"
                    image = Image.open(filename).convert('RGB')
                    image_input = processor(images=image, return_tensors="pt")

                    image_embed = model(**image_input).last_hidden_state
                    all_embeddings.append(image_embed)
                    store_num+=1
                else:
                    continue

                if (store_num + 1) % 300 == 0:
                    with open(f"{out_path_process}_{file_num}.pkl", 'wb') as f:
                        pickle.dump({"image_embedding": torch.cat(all_embeddings), "captions": all_captions}, f)

                    file_num += 1
                    all_embeddings = []
                    all_captions = []

            with open(f"{out_path_process}_{file_num}.pkl", 'wb') as f:
                pickle.dump({"image_embedding": torch.cat(all_embeddings), "captions": all_captions}, f)

        else:
            for i in tqdm(range(len(data['annotations']))):
                d = data['annotations'][i]
                img_id = d['image_id']
                filename = f"{data_path}/mini_{data_type}/COCO_{data_type}_{int(img_id):012d}.jpg"

                image = Image.open(filename).convert('RGB')
                image_input = processor(images=image, return_tensors="pt")

                image_embed = model(**image_input).last_hidden_state
                all_embeddings.append(image_embed)
                all_captions.append(d['caption'])
               
                if (i+1) % 300 == 0:
                    with open(f"{out_path_process}_{file_num}.pkl", 'wb') as f:
                        pickle.dump({"image_embedding":torch.cat(all_embeddings),"captions":all_captions}, f)

                    file_num += 1
                    all_embeddings = []
                    all_captions = []

            with open(f"{out_path_process}_{file_num}.pkl", 'wb') as f:
                pickle.dump({"image_embedding": torch.cat(all_embeddings), "captions": all_captions}, f)

        print(f'{data_type} Finish')

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_model_path', type=str, default='./models/clip-vit-base-patch16')
    parser.add_argument('--data_path',default='./data')
    args = parser.parse_args()
    exit(main(args.vision_model_path, args.data_path))
