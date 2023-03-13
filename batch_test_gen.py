import os

if __name__ == '__main__':
    batch_file = 'batch_test.bat'

    model_hiddens = [64,128,256]
    model_layers = [1,2,4]
    batch_sizes = [5,10]
    seeds = [42,1337]
    epochs = 10
    datatype = []
    learning_type = ['classification', 'segmentation']
    learning_rate = [0.1, 0.01, 0.001]
    datalength = -1
    os.remove(batch_file)

    for hidden in model_hiddens:
        for layer in model_layers:
            for batch_size in batch_sizes:
                for seed in seeds:
                    name = f'{hidden}_{layer}_{batch_size}_{seed}'
                    dir = os.path.join(r'C:\Users\Philipp\Desktop\MAThesis_GIT\MAThesis_AISponsorSkip\checkpoints',name)
                    execute_string = f'python main.py --cuda --hidden_size {hidden} --num_layers {layer} --bs {batch_size} --seed {seed} --checkpoint_dir {dir} --epochs {epochs} --datalen {datalength}'
                    with open(batch_file,mode='a') as f:
                        f.write(execute_string+'\n')