import os



if __name__ =='__main__':
    experiment_path = r'C:\temp\ModelSize_2nd_finished'
    dir_list =os.listdir(experiment_path)
    with open('test_only_batch.bat',mode='w') as f:
        for dir in dir_list:
            log = os.path.join(experiment_path,dir,'train.log')
            model = os.path.join(experiment_path,dir,'model009.t7')
            if os.path.isfile(log) and os.path.isfile(model):
                print(log,model)
                command = rf'python test_only.py --load_from {model} --log {log}'+'\n'
                f.write(command)