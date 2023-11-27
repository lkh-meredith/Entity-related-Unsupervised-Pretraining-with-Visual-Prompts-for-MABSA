from adapter_module import MappingNetwork
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer,AutoConfig,AutoModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import AutoModelForSeq2SeqLM
from torch.nn import CrossEntropyLoss
from model.modeling_t5 import T5ForConditionalGeneration,T5Model

import model.modeling_t5_casual as T5Casual

class FrozenLMT5DualMappingNetwork(nn.Module): #dual visual query
    def __init__(self, args):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_path)
        self.config = config
        self.constant_len = args.constant_len
        self.project_dim = args.mapping_dim
        self.embed_dim = config.hidden_size

        self.image_mapping_network = MappingNetwork(self.constant_len, self.embed_dim, self.project_dim)

        self.model = T5Casual.T5ForConditionalGeneration(config, args.freeze_lm, args.freeze_lm_head, self.constant_len)

        if args.freeze_MappingNetWork:
            for p in self.image_mapping_network.parameters():
                p.requires_grad_(False)

    def forward(self,
                image_features: torch.FloatTensor = None,
                input_ids: torch.FloatTensor = None,
                attention_mask: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: Optional[bool] = None,
                ):
        image_embeds = self.image_mapping_network(image_features)

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=input_ids.device)

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds = torch.cat([image_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        decoder_embeds = self.model.get_input_embeddings()(decoder_input_ids)
        decoder_embeds = torch.cat([image_embeds, decoder_embeds],dim=1)
        decoder_attention_mask = torch.cat([image_attention_mask,decoder_attention_mask],dim=1)

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds = decoder_embeds,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )

        return outputs["loss"]


class FrozenT5ForMABSADualQueries(nn.Module):
    def __init__(self,args):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_path)
        self.config = config
        self.constant_len = args.constant_len
        self.project_dim = args.mapping_dim
        self.embed_dim = config.hidden_size

        self.image_mapping_network = MappingNetwork(self.constant_len, self.embed_dim, self.project_dim)
        self.text_num_labels = 5

        self.model = T5Casual.T5Model(config, args.freeze_lm, self.constant_len)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.linear = nn.Linear(config.hidden_size, self.text_num_labels)
        self.loss_func = CrossEntropyLoss(ignore_index=-100)

        if args.freeze_MappingNetWork:
            for p in self.image_mapping_network.parameters():
                p.requires_grad_(False)
    

    def forward(self,
                image_features: torch.FloatTensor = None,
                input_ids: torch.FloatTensor = None,
                attention_mask: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: Optional[bool] = None,
                ):

        image_embeds = self.image_mapping_network(image_features)

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=input_ids.device)

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds = torch.cat([image_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        decoder_embeds = self.model.get_input_embeddings()(decoder_input_ids)
        decoder_embeds = torch.cat([image_embeds, decoder_embeds], dim=1)
        decoder_attention_mask = torch.cat([image_attention_mask, decoder_attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_embeds,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        features = outputs["last_hidden_state"]  

        features_droput = self.dropout(features)
        logits = self.linear(features_droput)  

        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.text_num_labels), labels.view(-1))

        return loss, logits

    def __init__(self,args):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_path)
        self.config = config
        self.constant_len = args.constant_len
        self.project_dim = args.mapping_dim
        self.embed_dim = config.hidden_size
        self.image_visual_mapping_network = MappingNetwork(self.constant_len, self.embed_dim, self.project_dim)
        self.text_num_labels = 5

        self.model = T5Model(config, args.freeze_lm, args.freeze_lm_head)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.linear = nn.Linear(config.hidden_size, self.text_num_labels) 
        self.loss_func = CrossEntropyLoss(ignore_index=-100)

        if args.freeze_MappingNetwork:
            for p in self.image_visual_mapping_network.parameters():
                p.requires_grad_(False)

    def forward(self,
                image_features: torch.FloatTensor = None,
                input_ids: torch.FloatTensor = None,
                attention_mask: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: Optional[bool] = None,
                ):

        image_embeds = self.image_visual_mapping_network(image_features)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=input_ids.device)
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds = torch.cat([image_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict

        )
        features = outputs["last_hidden_state"] 
        features_droput = self.dropout(features)
        logits = self.linear(features_droput)  

        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.text_num_labels), labels.view(-1))

        return loss, logits



