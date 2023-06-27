from transformers import AutoTokenizer,AutoModel,AutoConfig, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np


class BertMultiRegressionModel(nn.Module):
    def __init__(self, checkpoint, seed, label): 
        super(BertMultiRegressionModel,self).__init__() 

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.label = label

        #Load Model with given checkpoint and extract its body
        self.bert = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, \
                                            output_attentions=True, output_hidden_states=True, num_labels=1))
        self.dropout = nn.Dropout(0.1) 
        self.val_regressor = nn.Linear(768,1)
        self.act_regressor = nn.Linear(768,1)

    def forward(self, input_ids=None, attention_mask=None, valence=None, activation=None, call_id=None, sub_id=None, seg_id=None):
        embds = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(embds[0]) #outputs[0]=last hidden state
        val_logits = self.val_regressor(sequence_output[:,0,:].view(-1,768))
        act_logits = self.act_regressor(sequence_output[:,0,:].view(-1,768)) 
        
        # return val_logits, act_logits
        loss = None
        if valence == None and activation == None:
            return SequenceClassifierOutput(loss=loss, logits=[val_logits, act_logits])
        else:
            loss_fn = nn.MSELoss()
            if self.label=='both':     
                loss = torch.sqrt(loss_fn(val_logits.view(-1), valence.view(-1)) + loss_fn(act_logits.view(-1), activation.view(-1)))
            elif self.label=='valence':
                loss = torch.sqrt(loss_fn(val_logits.view(-1), valence.view(-1)))
            elif self.label=='activation':
                loss = torch.sqrt(loss_fn(act_logits.view(-1), activation.view(-1)))
            return SequenceClassifierOutput(loss=loss, logits=[val_logits, act_logits])
