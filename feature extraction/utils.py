import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, skew
import scipy.stats




def get_stats(l):
    # return the mean, std and various stats, given a list of values
    l = np.array(l)
    results = {}
    results['mean'] = np.mean(l) if len(l) else float('nan')
    results['std'] = np.std(l) if len(l) else float('nan')
    results['min'] = np.min(l) if len(l) else float('nan')
    results['max'] = np.max(l) if len(l) else float('nan')
    results['median'] = np.median(l) if len(l) else float('nan') 
    
#     x = np.array(range(len(l))).reshape(-1, 1)
#     lr = LinearRegression()
#     lr.fit(x, l)
#     results['lr_rs'] = lr.score(x, l)
#     results['lr_intercept'] = lr.intercept_
#     results['lr_coef'] = lr.coef_[0]
    return results



def get_stats_on_df(df, target_cols, group_by_col):
    # get a df of segments, convert it to a df of calls with the features (target_cols) calculated on 
    # segment statistics.
    # for example: get_stats_on_df(df_segs, ['valence', 'activation'], 'call_id')
    sample_stats_dict = {} #key: sample_id, value: a dict of obtained statistics
    samples = list(df[group_by_col].unique())
    for s in samples:
        results = {}
        df_t = df[df[group_by_col]==s]
        for c in target_cols:
            results = get_stats(list(df[c]), results, c)
        sample_stats_dict[s] = results.copy()
    df_r = pd.DataFrame.from_dict(sample_stats_dict, orient='index')
    return df_r


def z_normalize(df, target, mean, std):
    df[target+'_z'] = (df_target-mean)/std
    return df


# def bin_data_1d(x, mean=0, std=1):
#     bins = [-100, mean - 2*std, mean - std, mean - 0.5*std, mean, mean+0.5*std, mean + std, mean + 2*std, 100]
#     ret = stats.binned_statistic(x, None, 'count', bins=bins)
#     return ret.statistic
       
        
# def bin_data_2d(x, y, mean_x=0, mean_y=0, std_x=1, std_y=1):
#     x = np.array(x)
#     y = np.array(y)
#     binx = [-100, mean_x - 2*std_x, mean_x - std_x, mean_x, mean_x + std_x, mean_x + 2*std_x, 100]
#     biny = [-100, mean_y - 2*std_y, mean_y - std_y, mean_y - 0.5*std_y, mean_y, mean_y+0.5*std_y, mean_y + std_y, mean_y + 2*std_y, 100]
#     ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
#     # print(ret.statistic)
#     return ret.statistic


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


def depress_classify(x):
    if x<=7:
        return 0
    else:
        return 1
    
def mania_classify(x):
    if x<=9:
        return 0
    else:
        return 1
    
def get_corr_p(series1, series2):
    x = np.array(series1, dtype='float')
    y = np.array(series2, dtype='float')
    return scipy.stats.pearsonr(x, y)[1]


