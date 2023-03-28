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

special_tokens: list = []


def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) / 2))
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
            # tensored_targets = torch.zeros(len(data)).long()
            tensored_targets = torch.LongTensor(targets)
            # tensored_targets[torch.LongTensor(targets)] = 1
            tensored_targets = tensored_targets[:-1]
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue

    return batched_data, batched_targets, paths


def read_subtitle_entry(entry: list, index: int, word2vec, train: bool, return_w2v_tensors: bool = True, target_type: str = 'classification'):
    """
    Cleans the text, provides the corresponding targets and videoid of a given YouTube video's subtitles
    :param entry: subtitle entry for the YouTube video
    :param index: index of the subtitle entry, currently unused
    :param word2vec: Word2Vec model for vectorizing the words of the subtitles
    :param train: Currently unused
    :param return_w2v_tensors: Boolean value to return the words in a vectorized form or not
    :param target_type: either classification or segmentation, targets are processed either as 1 for inclass and 0 for notinclass
                    or as 0 for no change in segment and 1 on segment change
    :return: tuple of a list of the subtitle text, a list of the targets and the videoid as a str
    """
    segments = [segment[1] for segment in entry]
    video_id = entry[0][0]

    segment_count = 0
    new_text = []
    video_targets = []
    for idx, segment in enumerate(segments):
        words = extract_sentence_words(segment)
        if len(words) == 0:
            continue
        segment_count += 1
        if return_w2v_tensors:
            new_text.append([word_model(w, word2vec) for w in words])
        else:
            new_text.append(words)
        if target_type == 'classification':
            video_targets.append(entry[idx][2])
        elif target_type == 'segmentation':
            if idx == 0:
                video_targets.append(0)
            elif idx == len(segments) - 1:
                # 1 if last segment is sponsor, 0 if not
                # to differentiate between sponsor and beginning of next video in batch
                video_targets.append(entry[idx][2])
            else:
                if entry[idx - 1][2] != entry[idx][2]:
                    video_targets.append(1)
                else:
                    video_targets.append(0)

    return new_text, video_targets, video_id


# Returns a list of batch_size that contains a list of sentences, where each word is encoded using word2vec.
class SubtitlesDataset(Dataset):
    def __init__(self, db_path: str, word2vec, videoidlist: list, train: bool = False, target_type: str = 'classification',
                 subtitle_type: str = 'subtitles_db', execute_subtitles: list = None, max_segments: int = None):
        """
        Dataset class for the dataloader. Either uses provided subtitles
        or loads subtitles from the database from the provided list of video ids
        :param db_path: Path to sqlite database where the subtitles are stored
        :param word2vec: word2vec model for vectorizing the words in the subtitles
        :param videoidlist: list of video ids to be loaded from the database
        :param train: currently unused
        :param target_type: either classification or segmentation, targets are processed either as 1 for inclass and 0 for notinclass
                    or as 0 for no change in segment and 1 on segment change
        :param subtitle_type: either subtitles_db, generated_subtitles_db or execute. Determines where the data is loaded from
                        On execute subtitles have to be provided via the ex
        :param execute_subtitles: When using execute mode, provide subtitle segments as a list via this parameter
        :param max_segments: Maximum number of segments in a video, too long videos will not be processed
        """
        self.videoidlist = []
        self.target_type = target_type

        my_db = db.SponsorDB(db_path, no_setup=True)
        self.subtitles_list = []

        for videoid in videoidlist:
            if subtitle_type == 'subtitles_db':
                subtitles = my_db.get_subtitles_by_videoid(videoid)
            elif subtitle_type == 'generated_subtitles_db':
                subtitles = my_db.get_generated_subtitles_by_videoid(videoid)
            elif subtitle_type == 'execute':
                break
            else:
                raise ValueError(f'{subtitle_type} is not a valid subtitle type.')
            if max_segments is not None:
                if len(subtitles) <= max_segments:
                    self.subtitles_list.append(subtitles)
                    self.videoidlist.append(videoid)
            else:
                self.subtitles_list.append(my_db.get_subtitles_by_videoid(videoid))
                self.videoidlist.append(videoid)

        if subtitle_type == 'execute':
            if execute_subtitles is None:
                exit('No excute_subtitles provided')
            self.subtitles_list.append(execute_subtitles)
            self.videoidlist.append(execute_subtitles[0][0])

        self.train = train
        self.word2vec = word2vec

    def __getitem__(self, index):
        entry = self.subtitles_list[index]
        return read_subtitle_entry(entry, index, self.word2vec, self.train, self.target_type)

    def __len__(self):
        return len(self.subtitles_list)


if __name__ == '__main__':
    MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'
    my_db = db.SponsorDB(MY_DB_PATH)
    unique_videos = my_db.get_unique_video_ids_from_subtitles()
    unique_videos = unique_videos[:10]
    word2vec_temp = gensim.models.KeyedVectors.load_word2vec_format("word2vec/GoogleNews-vectors-negative300.bin",
                                                                    binary=True)
    test_DS = SubtitlesDataset(MY_DB_PATH, word2vec_temp, unique_videos)
    test_DL = DataLoader(test_DS, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for text, targets, id in test_DS:
        print(targets)

    for test in test_DL:
        print(test)

    exit(0)
