import pandas as pd
import numpy
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from driver import folds, runs
import torch



def fetch_data(df):
    # if ASR:
    # X = list(df['transcriptions_azure'])
    X = list(df['transcriptions'])
    y_val = list(df['text_valence'])
    y_act = list(df['text_activation'])
    sids = list(df['sub_id'])
    cids = list(df['call_id'])
    segids = list(df['seg_id'])
    ids = list(df['id'])
    return X, y_val, y_act, sids, cids, segids, ids


def get_train_dataset(use_asr, train_data_file, train_sid, val_sid):
    if use_asr:
        df = pd.read_csv(train_data_file)[['sub_id', 'call_id', 'seg_id', 'transcriptions_azure', 'text_activation', 'text_valence', 'id', 'confidence']]
        df = df[df['confidence']>=0][['sub_id', 'call_id', 'seg_id', 'id', 'transcriptions_azure', 'text_activation', 'text_valence']]
        df['text'] = df['transcriptions_azure']    
    else:    
        df = pd.read_csv(train_data_file)[['sub_id', 'call_id', 'seg_id', 'transcriptions', 'text_activation', 'text_valence', 'id', 'confidence']]
        df['text'] = df['transcriptions'].map(lambda x: str(x).replace('<comma>', \
                                                                    ', ').replace('<filler>', ''))
    df['text_activation'] = df['text_activation'].map(lambda x: (x-5)/4)
    df['text_valence'] = df['text_valence'].map(lambda x: (x-5)/4)
    df.rename(columns={'text_valence':'valence', 'text_activation':'activation'}, inplace=True)
    df_train = df[df['sub_id'].isin(train_sid)]
    df_val = df[df['sub_id'].isin(val_sid)]
    
    train_set = Dataset.from_pandas(df_train)
    val_set = Dataset.from_pandas(df_val)
    
    return train_set, val_set

def get_test_dataset(use_asr, test_data_file, test_sid):
    if use_asr:
        df = pd.read_csv(test_data_file)[['sub_id', 'call_id', 'seg_id', 'transcriptions_azure', 'text_activation', 'text_valence', 'id', 'confidence']]
        df = df[df['confidence']>=0][['sub_id', 'call_id', 'seg_id', 'id', 'transcriptions_azure', 'text_activation', 'text_valence', 'confidence']]
        df['text'] = df['transcriptions_azure']
    else:
        df = pd.read_csv(test_data_file)[['sub_id', 'call_id', 'seg_id', 'transcriptions', 'text_activation', 'text_valence', 'id', 'confidence']]
        df['text'] = df['transcriptions'].map(lambda x: str(x).replace('<comma>', \
                                                                    ', ').replace('<filler>', ''))
    df['text_activation'] = df['text_activation'].map(lambda x: (x-5)/4)
    df['text_valence'] = df['text_valence'].map(lambda x: (x-5)/4)
    df.rename(columns={'text_valence':'valence', 'text_activation':'activation'}, inplace=True)
    df_test = df[df['sub_id'].isin(test_sid)]

    test_set = Dataset.from_pandas(df_test)
    return test_set, df_test['id']

# How to access data:
# print(train_set[0])
# print(dtrain_set.data['label'])


def tokenize(batch, checkpoint='bert-base-uncased'):  
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer(batch["text"], truncation=True, max_length=512)


def tokenize_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch",columns=["input_ids", "attention_mask", "valence", "activation", 'sub_id', 'call_id', 'seg_id'])
    return dataset


def load_data(use_asr, train_data_file, test_data_file, run, folds, seed, batch_size=16, checkpoint='bert-base-uncased'):
    torch.manual_seed(seed)
    train_subjects = [folds[i] for i in run['train']]
    train_subjects = [sub for sublist in train_subjects for sub in sublist]
    val_subjects = folds[run['val']]
    test_subjects = folds[run['test']]

    train_set, val_set = get_train_dataset(use_asr, train_data_file, train_subjects, val_subjects)
    test_set, ids_test = get_test_dataset(use_asr, test_data_file, test_subjects)

    print('#Train:{}, #Eval:{}, #Test:{}'.format(len(train_set), len(val_set), len(test_set)))
 
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_set = tokenize_dataset(train_set)
    val_set = tokenize_dataset(val_set)
    test_set = tokenize_dataset(test_set)
    train_dataloader = DataLoader(train_set, batch_size, collate_fn=data_collator, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size, collate_fn=data_collator, shuffle=True)
    # print("sample id is ", test_set.data['sample_id'])
    test_dataloader = DataLoader(test_set, batch_size, collate_fn=data_collator, shuffle=False)
    # print("after Dataloader: ", test_dataloader.dataset[0])

    return train_dataloader, val_dataloader, test_dataloader, ids_test
