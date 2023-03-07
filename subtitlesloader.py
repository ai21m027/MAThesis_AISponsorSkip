import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import database as db
from text_manipulation import word_model, extract_sentence_words
import utils
import gensim
import math
import numpy as np

logger = utils.setup_logger(__name__, 'train.log')

special_tokens = []

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) /2))
    after_sentence_count = window_size - before_sentence_count - 1

    for data, targets, path in batch:
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window)))
            #tensored_targets = torch.zeros(len(data)).long()
            tensored_targets = torch.LongTensor(targets)
            #tensored_targets[torch.LongTensor(targets)] = 1
            tensored_targets = tensored_targets[:-1]
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue

    return batched_data, batched_targets, paths





def read_subtitle_entry(entry, index, word2vec, train, return_w2v_tensors=True):
    sentences = [segment[1] for segment in entry]
    video_id = entry[0][0]


    sentence_count = 0
    new_text =[]
    targets=[]
    for idx,sentence in enumerate(sentences):
        words = extract_sentence_words(sentence)
        if (len(words) == 0):
            continue
        sentence_count += 1
        if return_w2v_tensors:
            new_text.append([word_model(w, word2vec) for w in words])
        else:
            new_text.append(words)

        targets.append(entry[idx][2])

    return new_text, targets, video_id


# Returns a list of batch_size that contains a list of sentences, where each word is encoded using word2vec.
class SubtitlesDataset(Dataset):
    def __init__(self, db_path, word2vec, videoidlist: list, train=False):
        # self.my_db_path = db_path
        self.videoidlist = videoidlist

        my_db = db.SponsorDB(db_path, no_setup=True)
        self.subtitles_list = []
        for videoid in videoidlist:
            self.subtitles_list.append(my_db.get_subtitles_by_videoid(videoid))

        self.train = train
        self.word2vec = word2vec

    def __getitem__(self, index):
        entry = self.subtitles_list[index]
        # returns text, targets, video_id
        return read_subtitle_entry(entry, index, self.word2vec, self.train)

    def __len__(self):
        return len(self.videoidlist)


if __name__ == '__main__':
    MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'
    my_db = db.SponsorDB(MY_DB_PATH)
    unique_videos = my_db.get_unique_video_ids_from_subtitles()
    unique_videos = unique_videos[:10]
    word2vec = gensim.models.KeyedVectors.load_word2vec_format("word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    test_DS = SubtitlesDataset(MY_DB_PATH, word2vec, unique_videos)
    test_DL = DataLoader(test_DS, batch_size=8, shuffle=True,collate_fn=collate_fn)
    for text,targets,id in test_DS:
        print(targets)


    for test in test_DL:
        print(test)

    exit(0)