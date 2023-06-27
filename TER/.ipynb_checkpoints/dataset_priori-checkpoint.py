import pandas as pd
import numpy
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset

def fetch_data(df):
    X = list(df['transcriptions'])
    y_val = list(df['valence'])
    y_act = list(df['activation'])
    return X, y_val, y_act


def get_dataset(data_file, train_sid, val_sid, test_sid):
    df = pd.read_csv(data_file)[['sub_id', 'transcriptions', 'activation', 'valence']]
    df['transcriptions'] = df['transcriptions'].map(lambda x: str(x).replace('<comma>', ', ').replace('<filler>', ''))
    df['activation'] = df['activation'].map(lambda x: (x-4)/5)
    df['valence'] = df['valence'].map(lambda x: (x-4)/5)

    df_train = df[df['sub_id'].isin(train_sid)]
    X_train, y_val_train, y_act_train = fetch_data(df_train)
    df_val = df[df['sub_id'].isin(val_sid)]
    X_val, y_val_val, y_act_val = fetch_data(df_val)
    df_test = df[df['sub_id'].isin(test_sid)]
    X_test, y_val_test, y_act_test = fetch_data(df_test)

    train_df = pd.DataFrame.from_dict({'text':X_train, 'valence':y_val_train, 'activation':y_act_train})
    val_df = pd.DataFrame.from_dict({'text':X_val, 'valence':y_val_val, 'activation':y_act_val})
    test_df = pd.DataFrame.from_dict({'text':X_test, 'valence':y_val_test, 'activation':y_act_test})
    train_set = Dataset.from_pandas(train_df)
    val_set = Dataset.from_pandas(val_df)
    test_set = Dataset.from_pandas(test_df)
 
    return train_set, val_set, test_set

# How to access data:
# print(train_set[0])
# print(dtrain_set.data['label'])