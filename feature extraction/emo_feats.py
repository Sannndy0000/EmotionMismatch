from utils import *
import numpy as np


# those df inputs are df with all segments from one call
def get_emo_feats(df, feats={}):
    for c in ['text_valence', 'text_activation', 'acoustic_valence', 'acoustic_activation']:
    # for c in ['acoustic_valence', 'acoustic_activation']:
        results = get_stats(df[c])
        for k in results.keys():
            feats['EMO_'+c+'_'+k] = results[k]
    return feats
        

def get_emo_cov(df, feats={}):
    # time series
    text_val = np.array(df['text_valence'])
    text_act = np.array(df['text_activation'])
    acou_val = np.array(df['acoustic_valence'])
    acou_act = np.array(df['acoustic_activation'])
    text_mat = np.cov(text_val,text_act)
    acou_mat = np.cov(acou_val,acou_act)
    feats['EMO_'+'text_val_var'] = text_mat[0][0]
    feats['EMO_'+'text_act_var'] = text_mat[1][1]
    feats['EMO_'+'acou_val_var'] = acou_mat[0][0]
    feats['EMO_'+'acou_act_var'] = acou_mat[1][1]
    feats['EMO_'+'text_cov'] = text_mat[0][1]
    feats['EMO_'+'acou_cov'] = acou_mat[0][1]
    return feats
    
    

    
