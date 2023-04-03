from os import walk
import pandas as pd
from typing import NamedTuple
import re
import numpy as np
import os
class Experiment(NamedTuple):
    epoch: int
    hidden: int
    layers: int
    subtitle_type: str
    type:str
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

def parse_log(path: str,dir:str,is_eval=False) -> Experiment:
    params = dir.split('_')
    #hidden_size = int(params[0])
    #layers = int(params[1])
    train_log=os.path.join(os.path.split(path)[0],'train.log')
    if is_eval:
        with open(train_log, 'r') as f:
            lines = f.readlines()
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
    found = False
    lr = float(re.search("'lr': (\d.\d+)",lines[0]).group(1))
    seed = int(re.search("'seed': (\d+)",lines[0]).group(1))
    subtitle_type = re.search("'subtitletype': '(\w+)'",lines[0]).group(1) if re.search("'subtitletype': '(\w+)'",lines[0]) is not None else 'manual'
    epoch = int(re.search("'epochs': (\d+)",lines[0]).group(1))
    hidden = int(re.search("'hidden_size': (\d+)",lines[0]).group(1))
    num_layers = int(re.search("'num_layers': (\d+)",lines[0]).group(1))
    type = re.search("'type': '(\w+)'", lines[0]).group(1) if re.search("'type': '(\w+)'",lines[0]) is not None else 'classification'
    if is_eval:
        with open(path, 'r') as f:
            lines = f.readlines()
    for idx,line in enumerate(lines):
        if re.search(r'Validating Epoch', line):
            accuracy = float(re.search(r'accuracy: (\d+\.\d+),', line).group(1))
            pk = float(re.search(r'Pk: (\d+\.\d+),', line).group(1))
            windiff = float(re.search(r'Windiff: (-?\d+\.?\d*),', line).group(1))
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
        result = Experiment(epoch=epoch,
                            hidden=hidden,
                            layers=num_layers,
                            subtitle_type = subtitle_type,
                            type=type,
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
    experiment_base_dir = r'C:\temp\ModelSize_2nd_finished'
    pd.set_option('display.max_columns',None)
    experiments = []
    for root, dirs, files in walk(experiment_base_dir):
        for file in files:
            if file.endswith('eval.log'):
                #print(dir,file)
                test = root
                if root.split('\\')[-1] != r'checkpoints':
                    experiments.append(parse_log(f'{root}\\{file}',root.split('\\')[-1],is_eval=True))

    experiments_df = pd.DataFrame(experiments)
    #experiments_df.columns = ['epoch','hidden','layers','seed','Acc','Pk','Windiff','F1','Loss','TN','FN','FP','TP']

    print(experiments_df)
    #print(experiments_df.describe())
    experiments_df.to_csv('results.csv',sep=';',decimal= ",")
