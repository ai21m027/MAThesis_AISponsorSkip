from os import walk
import pandas as pd
from typing import NamedTuple
import re
import numpy as np

class Experiment(NamedTuple):
    epoch: int
    hidden: int
    layers: int
    seed: int
    lr: float
    accuracy: float
    pk: float
    windiff: float
    f1: float
    loss: float
    tn: int
    fn: int
    fp: int
    tp: int

def parse_log(path: str,dir:str) -> Experiment:
    params = dir.split('_')
    #hidden_size = int(params[0])
    #layers = int(params[1])
    lr = float(params[0])
    seed = int(params[1])
    with open(path, 'r') as f:
        lines = f.readlines()
    found = False
    for idx,line in enumerate(lines):
        if re.search(r'Validating Epoch', line):
            accuracy = float(re.search(r'accuracy: (\d+\.\d+),', line).group(1))
            pk = float(re.search(r'Pk: (\d+\.\d+),', line).group(1))
            windiff = float(re.search(r'Windiff: (-\d+\.\d+),', line).group(1))
            f1 = float(re.search(r'F1: (\d+\.\d+) ,', line).group(1))
            loss = float(re.search(r'Loss: (\d+\.\d+)', line).group(1))
            found = True
            next_line = lines[idx+1]
            tn = int(re.search('TN: (\d+) ',next_line).group(1))
            fn = int(re.search('FN: (\d+) ',next_line).group(1))
            fp = int(re.search('FP: (\d+) ',next_line).group(1))
            tp = int(re.search('TP (\d+)',next_line).group(1))

    if found:
        #print(accuracy, pk, windiff, f1, loss)
        result = Experiment(epoch=20,
                            hidden=256,
                            layers=2,
                            seed=seed,
                            lr=lr,
                            accuracy=accuracy,
                            pk=pk,
                            windiff=windiff,
                            f1=f1,
                            loss=loss,
                            tn=tn,
                            fn=fn,
                            fp=fp,
                            tp=tp)
        return result
    else:
        return None

if __name__ == '__main__':
    experiment_base_dir = 'checkpoints'
    pd.set_option('display.max_columns',None)
    experiments = []
    for root, dirs, files in walk(experiment_base_dir):
        for dir in dirs:
            for file in files:
                if file.endswith('train.log'):
                    #print(dir,file)

                    experiments.append(parse_log(f'{experiment_base_dir}/{dir}/{file}',dir))

    experiments_df = pd.DataFrame(experiments)
    #experiments_df.columns = ['epoch','hidden','layers','seed','Acc','Pk','Windiff','F1','Loss','TN','FN','FP','TP']

    print(experiments_df)
    #print(experiments_df.describe())
    experiments_df.to_csv('results.csv',sep=';',decimal= ",")
    model_hiddens = [64, 128, 256, 512]
    model_layers = [1, 2, 4, 6]
    lrs = [0.1,0.01,0.001,0.0001]
    experiments_dict ={}
    for lr in lrs:
        name = f'{lr}'

        for experiment in experiments:
            if experiment is not None:
                if experiment.lr == lr:
                    if name not in experiments_dict:
                        experiments_dict[name] = []
                    experiments_dict[name].append(np.array(list(experiment)))
        if name in experiments_dict:
            new_list =np.mean(experiments_dict[name],axis=0)
            experiments_dict[name] = new_list

    avg_data = pd.DataFrame(experiments_dict)
    avg_data = avg_data.T
    #avg_data.columns = ['epoch','hidden','layers','seed','Acc','Pk','Windiff','F1','Loss','TN','FN','FP','TP']
    avg_data.columns = experiments_df.columns
    print(avg_data)
    avg_data.to_csv('results_avg.csv',sep=';',decimal= ",")