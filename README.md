# Entity-related-Unsupervised-Pretraining-with-Visual-Prompts-for-MABSA
The code of our paper "Entity-related Unsupervised Pretraining with Visual Prompts for Multimodal Aspect-based Sentiment Analysis"

## Data Download
The MABSA dataset can be derived from the paper: Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis (https://github.com/NUSTM/VLP-MABSA)

The pre-training dataset can download from the COCO2014: https://cocodataset.org/

The [split_coco.py](https://github.com/lkh-meredith/Entity-related-Unsupervised-Pretraining-with-Visual-Prompts-for-MABSA/blob/main/split_coco.py) is used to split COCO2014 for pre-training.

## Data pre-process
We use [clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) to extract image feature. 

[parse_coco.py](https://github.com/lkh-meredith/Entity-related-Unsupervised-Pretraining-with-Visual-Prompts-for-MABSA/blob/main/parse_coco.py) and [parse_twitter.py](https://github.com/lkh-meredith/Entity-related-Unsupervised-Pretraining-with-Visual-Prompts-for-MABSA/blob/main/parse_twitter.py) is used to pre-process data. 

## Model backbone
We use [flan-t5-base](https://huggingface.co/google/flan-t5-base) and [t5-base](https://huggingface.co/t5-base) to initialize our model.
