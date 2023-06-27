import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy import stats
from utils import get_stats


def get_abs_mm(df_c, feats):
    val_abs_mm = get_stats(list(df_c['mm_valence']))
    act_abs_mm = get_stats(list(df_c['mm_activation']))
    dist_mm = get_stats(list(df_c['mm_distance']))
    for stat in ['mean', 'std', 'min', 'max', 'median']: #, 'qntl_25', 'qntl_75']:
        feats['MM_val_abs_'+stat] = val_abs_mm[stat]
        feats['MM_act_abs_'+stat] = act_abs_mm[stat]
        feats['MM_dist_'+stat] = dist_mm[stat]  
    return feats


def get_inter_mm(df_c, feats):
    mm_val_inter_text_val = np.multiply(df_c['mm_valence'], df_c['text_valence'])
    mm_val_inter_text_act = np.multiply(df_c['mm_valence'], df_c['text_activation'])
    mm_act_inter_text_val = np.multiply(df_c['mm_activation'], df_c['text_valence'])
    mm_act_inter_text_act = np.multiply(df_c['mm_activation'], df_c['text_activation'])
    name2list = {
                 'mm_val_inter_text_val':mm_val_inter_text_val, 
                 'mm_val_inter_text_act':mm_val_inter_text_act, 
                 'mm_act_inter_text_val':mm_act_inter_text_val, 
                 'mm_act_inter_text_act':mm_act_inter_text_act
    }
    
    for name, item in name2list.items():
        stats_dict = get_stats(item)
        for stat in ['mean', 'std', 'min', 'max', 'median']: #, 'qntl_25', 'qntl_75']:
            feats['MM_'+name+'_'+stat] = stats_dict[stat] 
    return feats
 
    
def normalize_mm(df_c, df, targets = ['mm_activation', 'mm_valence', 'mm_distance']):
    df_euthy = df[df['mood']==0]  
    test_sub = list(df_c['sub_id'].unique())[0]
    df_euthy = df_euthy[df_euthy['sub_id']!=test_sub]
    for t in targets:
        z_mean = df_euthy[t].mean()
        z_std = df_euthy[t].std()
        df_c[t+'_z'] = (df_c[t]-z_mean)/z_std 
    return df_c



def get_cov_mm(df, feats={}):
    # time series
    text_val = np.array(df['text_valence'])
    text_act = np.array(df['text_activation'])
    acou_val = np.array(df['acoustic_valence'])
    acou_act = np.array(df['acoustic_activation'])
    feats['MM_text_val_acou_val_cov'] = np.cov(text_val,acou_val)[0][1]
    feats['MM_text_val_acou_act_cov'] = np.cov(text_val,acou_act)[0][1]
    feats['MM_text_act_acou_val_cov'] = np.cov(text_act,acou_val)[0][1]
    feats['MM_text_act_acou_act_cov'] = np.cov(text_act,acou_act)[0][1]

    return feats