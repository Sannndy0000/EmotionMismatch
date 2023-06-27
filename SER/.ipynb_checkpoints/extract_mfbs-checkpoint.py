import os
import numpy as np
import pandas as pd 
import librosa
from python_speech_features import logfbank
from joblib import Parallel, delayed
from IPython import embed

SAMPLE_RATE=8000
MFB_DIM = 40 
MFB_WIN_LEN = 0.025
MFB_WIN_STEP= 0.01
MFB_NFFT = 2048 
MFB_LOW_FREQ = 0 
MFB_HIGH_FREQ = None
MFB_PREEMPH = 0.97 


def extract_mfb(input_file, output_file):

        if isinstance(input_file, str): 
            x, sr = librosa.load(input_file, sr=SAMPLE_RATE)
        else: 
            x, sr = input_file

        # extract 
        x = logfbank(x, 
                samplerate=sr,
                winlen=MFB_WIN_LEN,
                winstep=MFB_WIN_STEP,
                nfilt=MFB_DIM,
                nfft=MFB_NFFT,
                lowfreq=MFB_LOW_FREQ,
                highfreq=MFB_HIGH_FREQ,
                preemph=MFB_PREEMPH)

        np.save(output_file, arr=np.transpose(x))

def segment_extract_mfbs(row):
        
    input_dir = '/nfs/turbo/McInnisLab/priori_v1_data/call_audio/speech/'
    output_dir = '/nfs/turbo/McInnisLab/sandymn/ICASSP2023_mismatch/mismatch-journal/data/full_priori/data_full/mfbs_personal/' 
    
    input_file = os.path.join(input_dir, str(row['call_id']) + '.wav')
    output_file = os.path.join(output_dir, str(row['sub_id']) + '_' + str(row['call_id']) + '_' + str(row['seg_num']) + '.npy')
    
    if os.path.exists(output_file):
        return 

    x, sr = librosa.load(input_file, sr=SAMPLE_RATE)
    x = x[int(row['begin_time'] * sr): int((row['begin_time'] + row['duration']) * sr)]
    extract_mfb((x, sr), output_file)    




if __name__ == '__main__': 


    # use data_list = 'df' for priori emotion 
    # use data_list = 'dir' for priori 
    data_list = 'df' 

    if data_list == 'df': 
        df = pd.read_csv('/nfs/turbo/McInnisLab/sandymn/brown_v1_segments/data/old/segments.csv') 
        df['output'] = '/nfs/turbo/McInnisLab/sandymn/brown_v1_segments/data/old/mfbs/' + df['id'] + '.npy'
        df = df[['path','output']]

        Parallel(n_jobs=8)(delayed(extract_mfb)(row['path'], row['output']) for _, row in df.iterrows())
    
    if data_list == 'dir':
        metadata_dir = '/nfs/turbo/McInnisLab/sandymn/ICASSP2023_mismatch/mismatch-journal/data/full_priori/data_full/data_personal/'
        metadata_files = os.listdir(metadata_dir)
        df_list = [] 
        for f in metadata_files: 
            if 'checkpoint' not in f:
                df_list.append(pd.read_csv(os.path.join(metadata_dir, f), index_col=0))
        df = pd.concat(df_list)

        Parallel(n_jobs=8)(delayed(segment_extract_mfbs)(row) for _, row in df.iterrows())
