import os, sys, argparse

from dataset_priori import *
from models import *
# from test import test
from train import *
from dataset_priori import *
import torch


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

seeds = [1, 2, 3, 4, 5]
checkpoint='bert-base-uncased'

def main(args):
    print(args.use_asr)
    results = []
    for seed in seeds:
        for run in runs:
            print(run)
            print('Train: {}, Test: {}'.format(args.train_data, args.test_data))
            print('Optimize on', args.label)
            train_data_file = 'train_priori_emotion.csv' # change to your own path
            test_data_file = 'test_priori_emotion.csv'
            train_loader, val_loader, test_loader, ids_test = load_data(args.use_asr, train_data_file, test_data_file, run, folds, seed)
            
            best_model = train(train_loader, val_loader, args.label, checkpoint, args.lr, seed)
            test_result = test(best_model, test_loader, args.label)
            test_result['seed'] = seed
            test_result['id'] = ids_test
            results.append(test_result)
            
            result_df = pd.concat(results)
            if args.use_asr:
                torch.save(best_model.state_dict(), 'models/model_{}_ASR_{}_{}.pt'.format(args.train_data, run['test'], seed))
                result_df.to_csv('{}results_asr_{}_{}_{}.csv'.format(args.exp_dir, args.train_data, args.test_data, args.label))

            else:
                torch.save(best_model.state_dict(), 'models/model_{}_{}_{}.pt'.format(args.train_data, run['test'], seed))
                result_df.to_csv('{}results_{}_{}_{}.csv'.format(args.exp_dir, args.train_data, args.test_data, args.label))


if __name__=='__main__':
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()

    # main arguments for setting up experiment
    parser.add_argument('--train_data', type=str, default='both',
                        help='assess, personal or both,   \
                             will be used to save models, logs, and results')
    parser.add_argument('--test_data', type=str, default='both',
                        help='assess, personal or both,   \
                             will be used to save models, logs, and results')
    parser.add_argument('--label', type=str, default='both',
                        help='Label to train on: activation, valence, or both')

    parser.add_argument('--exp_dir', type=str,
                        help='Path to models, logs, and results folders')
    parser.add_argument('--use_asr', type=bool, default=False,
                        help='Use ASR trasncriptions instead of manual transcriptions.')
    parser.add_argument('--lr', type=float, default=5e-6)

    args = parser.parse_args()



    main(args)