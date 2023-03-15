import os

if __name__ == '__main__':
    batch_file = 'batch_test.bat'

    model_hiddens = [64,128,256,512]
    model_layers = [1,2,4,6]
    #batch_sizes = [5,10]
    lrs = [0.01,0.001,0.0001]
    seeds = [42,1337,1234,4321]
    epochs = 10
    datatype = []
    learning_type = ['classification', 'segmentation']
    datalength = 10000
    os.remove(batch_file)

    for hidden in model_hiddens:
        for layer in model_layers:
            #for lr in lrs:
            for seed in seeds:
                #name = f'{hidden}_{layer}_{lr}_{seed}'
                name = f'{hidden}_{layer}_{seed}'
                dir = os.path.join(r'C:\Users\Philipp\Desktop\MAThesis_GIT\MAThesis_AISponsorSkip\checkpoints',name)
                execute_string = f'python main.py --cuda --hidden_size {hidden} --num_layers {layer} --bs {5} --seed {seed} --checkpoint_dir {dir} --epochs {epochs} --datalen {datalength}'
                with open(batch_file,mode='a') as f:
                    f.write(execute_string+'\n')