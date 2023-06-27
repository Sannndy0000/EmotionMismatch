import numpy as np
from scipy import stats


def CCC(y_true, y_pred):
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    denominator=var_true+var_pred+(mean_true-mean_pred)**2
    return numerator/denominator

def RMSE(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def compute_metric_single(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = RMSE(y_true, y_pred)
    ccc = CCC(y_true, y_pred)
    pcc = stats.pearsonr(y_true, y_pred)[0]
    
    print("PCC:", pcc, '\tCCC:', ccc, '\tRMSE:', rmse)
    return {"PCC":pcc, 'CCC':ccc, 'RMSE':rmse}

def compute_metric(val_true, act_true, val_pred, act_pred):
    val_metric = np.nan
    act_metric = np.nan
    # if val_true != None:
    print('Valence')
    val_metric = compute_metric_single(val_true, val_pred)
    # if act_true != None:
    print('Activation')
    act_metric = compute_metric_single(act_true, act_pred)
    return val_metric, act_metric


def score2mood(hamd, ymrs):
    if hamd<=7 and ymrs <= 9:
        return 0
    if ymrs<=9 and 8<=hamd<=16:
        return -1
    if hamd>=17:
        return -2
    if hamd<=7 and 10<=ymrs<=20:
        return 1
    if ymrs>=21:
        return 2
    else:
        return -1
    
def add_mood_to_df(df):
    mood = []
    for i, row in df.iterrows():
        mood.append(score2mood(row['depression_rating'], row['mania_rating']))
    df['mood'] = mood 
    return df