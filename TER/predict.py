import pandas as pd
import torch
import numpy
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import os
from dataset_priori import *
from models import *

# folds = {0: [*fold_1_subjects*],
#          ...
#          8: [*fold_8_subjects*]}

runs = [{'test': 0, 'val': 1, 'train': [2, 3, 4, 5, 6, 7, 8]},
        {'test': 1, 'val': 2, 'train': [0, 3, 4, 5, 6, 7, 8]},
        {'test': 2, 'val': 3, 'train': [0, 1, 4, 5, 6, 7, 8]},
        {'test': 3, 'val': 4, 'train': [0, 1, 2, 5, 6, 7, 8]},
        {'test': 4, 'val': 5, 'train': [0, 1, 2, 3, 6, 7, 8]},
        {'test': 5, 'val': 6, 'train': [0, 1, 2, 3, 4, 7, 8]},
        {'test': 6, 'val': 7, 'train': [0, 1, 2, 3, 4, 5, 8]},
        {'test': 7, 'val': 8, 'train': [0, 1, 2, 3, 4, 5, 6]},
        {'test': 8, 'val': 0, 'train': [1, 2, 3, 4, 5, 6, 7]}]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model(seed, run, root_path):
    model = BertMultiRegressionModel('bert-base-uncased', seed, 'both')
    model_path = root_path + 'model_both_{}_{}.pt'.format(run, seed)
    model.load_state_dict(torch.load(model_path))
    return model


def tokenize_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch",columns=["input_ids", "attention_mask"])
    return dataset


def get_pred_dataset(pred_data_file, batch_size=16, checkpoint='bert-base-uncased'):
    df = pd.read_csv(pred_data_file)# [['subject_id', 'call_id', 'seg_num', 'text']]

    df.rename(columns={'seg_num':'seg_id', 'subject_id':'sub_id'}, inplace=True)
    pred_set = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    pred_set = tokenize_dataset(pred_set)
    pred_dataloader = DataLoader(pred_set, batch_size, collate_fn=data_collator, shuffle=False)
    return pred_dataloader, df


def get_pred_dataset_brown(pred_data_file, batch_size=16, checkpoint='bert-base-uncased'):
    df = pd.read_csv(pred_data_file)# [['subject_id', 'call_id', 'seg_num', 'text']]

    df.rename(columns={'subject_id':'sub_id'}, inplace=True)
    pred_set = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    pred_set = tokenize_dataset(pred_set)
    pred_dataloader = DataLoader(pred_set, batch_size, collate_fn=data_collator, shuffle=False)
    return pred_dataloader, df


def predict(model, pred_loader):
    model = model.to(device)
    model.eval()
    val_logits = []
    act_logits = []
    for batch in pred_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        val_logits.extend(outputs.logits[0].cpu().view(-1).tolist())
        act_logits.extend(outputs.logits[1].cpu().view(-1).tolist())
  

    # results = pd.DataFrame({'subject': subjects, 'call_id':call_ids, 'seg_id':seg_ids, \
    #                                 'text_valence': val_logits, 'text_activation': act_logits})
    return val_logits, act_logits




def emo_pred(filename, root_path, new_path):   
    # (5 seeds * runs) * num_samples
    valences = []
    activations = []
    pred_dataloader, pred_df = get_pred_dataset(root_path + filename)
    for seed in [1, 2, 3]:
        for run in range(3):# should be 9 if use full
            # if int(filename[7:-4]) in folds[runs[run]['test']]:
            #     continue
            # print("-------------------- SEED: {}, RUN: {} --------------------".format(seed, run))
            model = load_model(seed, run) 
            val_logits, act_logits= predict(model, pred_dataloader)
            valences.append(val_logits)
            activations.append(act_logits)
    valences = np.array(valences)
    valences = np.mean(valences, axis=0)
    activations = np.array(activations)
    activations = np.mean(activations, axis=0)
    pred_df['text_velence'] = valences*4+5
    pred_df['text_activation'] = activations*4+5
    pred_df.to_csv(new_path + filename)
    # print(valences.shape, activations.shape)
    

if __name__=='__main__':

    # if a directory

    # root_path = 'path_to_data'
    # new_path = 'path_to_write_output'
    # for i, filename in enumerate(os.listdir(root_path)):
    #     print(i, filename)
    #     if os.path.isfile(os.path.join(new_path, filename)):
    #         continue
    #     f = os.path.join(root_path, filename)
    #     if os.path.isfile(f):
    #         emo_pred(filename, root_path, new_path)


    # if a single file

    old_path = 'path_to_data'
    new_path = 'path_to_write_output'
    filename = 'segments.csv'
    emo_pred(filename, old_path, new_path)

    




