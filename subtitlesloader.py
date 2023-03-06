from pathlib2 import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from text_manipulation import split_sentences, word_model, extract_sentence_words
import utils
import math



logger = utils.setup_logger(__name__, 'train.log')



# Returns a list of batch_size that contains a list of sentences, where each word is encoded using word2vec.
class SubtitlesDataset(Dataset):
    def __init__(self, root, word2vec, train=False, folder=False,manifesto=False, folders_paths = None):
        self.manifesto = manifesto
        if folders_paths is not None:
            self.textfiles = []
            for f in folders_paths:
                self.textfiles.extend(list(f.glob('*.ref')))
        elif (folder):
            self.textfiles = get_choi_files(root)
        else:
            self.textfiles = list(Path(root).glob('**/*.ref'))

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_choi_file(path, self.word2vec, self.train,manifesto=self.manifesto)

    def __len__(self):
        return len(self.textfiles)
