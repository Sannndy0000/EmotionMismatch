from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import scipy
import os
from nltk import word_tokenize
import pickle

def generate_vectorizer(df_c):
#     texts = []
#     path = '../mismatch-journal/data/full_priori/data_r/'
#     for filename in os.listdir(path):
#         f = os.path.join(path, filename)
#         if os.path.isfile(f):
#             df = pd.read_csv(f)[['call_id', 'seg_id', 'text', 'is_assessment']]
#             df = df[df['is_assessment']=='t']
#             # df = df[df.groupby('call_id')['call_id'].transform('size') >= 5]
#             df = df[['call_id','text']].groupby(['call_id'])['text'].apply(lambda x: '\n'.join(x)).reset_index()

#             df['wc'] = df['text'].map(lambda x: len(word_tokenize(x)))
#             df = df[df['wc']>=50]
#             # print(len(df))
#             texts.extend(df['text'])
    texts = df_c['transcriptions']
    # texts = pickle.load(open('texts.pk', 'rb'))
    vectorizer_1 = TfidfVectorizer(ngram_range=(1, 1), min_df = 10, stop_words='english')#, norm=None)
    vectorizer_2 = TfidfVectorizer(ngram_range=(2, 2), min_df = 50, stop_words='english')#, norm=None)
    vectorizer_1 = vectorizer_1.fit(texts)
    vectorizer_2 = vectorizer_2.fit(texts)
    unigrams = vectorizer_1.get_feature_names_out()
    bigrams = vectorizer_2.get_feature_names_out()
    ngrams = list(unigrams)+list(bigrams)
    print(len(unigrams)+len(bigrams))
    return vectorizer_1, vectorizer_2

def get_tfidf_feats(df_c, vect1, vect2):
    texts = list(df_c['transcriptions'])
    lengths = np.array([len(word_tokenize(t)) for t in texts])
    X_1 = vect1.transform(texts).todense()# /lengths[:, None]
    X_2 = vect2.transform(texts).todense()# /lengths[:, None]
    X = np.concatenate((X_1, X_2), axis=1)
    X = np.concatenate((X_1, X_2), axis=1)
    # print(X_1.shape, X_2.shape, X.shape)
    unigrams = vect1.get_feature_names_out()
    bigrams = vect2.get_feature_names_out()
    ngrams = list(unigrams)+list(bigrams)
    return X, ngrams


def select_tfidf_feats(X, ngrams, hamd, ymrs, target, threshold=0.01):
    labels = None
    if 'depress' in target:
        labels = hamd
    elif 'mania' in target:
        labels = ymrs
    cols_keep = []
    if 'mood' in target:
        for i in range(len(ngrams)):
            X_i = np.array(X[:, i]).flatten()
            if min(scipy.stats.pearsonr(X_i, hamd)[1], scipy.stats.pearsonr(X_i, ymrs)[1]) < threshold:
                    cols_keep.append(i)
    else:
        for i in range(len(ngrams)):
            X_i = np.array(X[:, i]).flatten()
            if scipy.stats.pearsonr(X_i, labels)[1] < threshold:
                    cols_keep.append(i)
        
    ngrams = [ngrams[i] for i in cols_keep]
    X = X[:, cols_keep]
    # print('{} tfidf features kept.'.format(len(cols_keep)))
    #print(ngrams)
    return X, ngrams, cols_keep


def add_tfidf_to_test(df_test, vect1, vect2, tfidf_keep, ngrams):
    texts = list(df_test['transcriptions'])
    lengths = np.array([len(word_tokenize(t)) for t in texts])
    X_1 = vect1.transform(texts).todense()# /lengths[:, None]
    X_2 = vect2.transform(texts).todense()# /lengths[:, None]
    X = np.concatenate((X_1, X_2), axis=1)
    X = X[:, tfidf_keep]
    for i, name in enumerate(ngrams):
        df_test[name] = np.array(X[:, i])
    return df_test
             
    
    
# tfidf_keep = []
#     vectorizer = None
#     ngrams = []
#     if include_tfidf:
#         X, ngrams, vectorizer = get_tfidf_feats(df['transcriptions'])
#         X, ngrams, tfidf_keep = select_tfidf_feats(X, ngrams, df['hamd'], df['ymrs'], target)
#         for i, name in enumerate(ngrams):
#             df[name] = np.array(X[:, i])
#             cols_keep.append(name)