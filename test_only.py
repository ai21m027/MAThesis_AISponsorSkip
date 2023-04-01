from argparse import ArgumentParser, Namespace
from typing import NamedTuple
import re
import os
import utils
from pathlib2 import Path
import logging
from utils import maybe_cuda
import torch
import database as db
import random
import gensim
import numpy as np
import accuracy
from subtitlesloader import SubtitlesDataset, collate_fn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from nltk.metrics import windowdiff, pk
from logging import Logger

preds_stats = utils.predictions_analysis()
MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'


class Experiment_parameters(NamedTuple):
    hidden: int
    layers: int
    subtitle_type: str
    seed: int
    lr: float
    datalen:int
    type: str
    max_segment_number:int

class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def convert_class_to_seg(output_seg):
    output = []
    for idx, out in enumerate(output_seg):
        if idx == 0:
            output.append(out)
        elif idx == len(output_seg) - 1:
            # 1 if last sentence is sponsor, 0 if not
            # to differentiate between sponsor and beginning of next video in batch
            output.append(out)
        else:
            if output_seg[idx - 1] != out:
                output.append(1)
            else:
                output.append(0)
    return output


def test(model: torch.nn.Module, args: Namespace, epoch: int, dl: DataLoader, logger: Logger) -> (
        float, float):
    """
    Tests a ML model on the given dataloader
    :param model: The ML model to be validated
    :param args: config args from the commandline
    :param epoch: last finished epoch before evaluation
    :param dl: Dataloader for validation data
    :param logger: Logger instance
    :return: pk value of validation, windiff value of validation
    """
    model.eval()
    with tqdm(desc='Validating', total=len(dl)) as pbar:
        acc = Accuracies()
        total_loss = float(0)
        total_out = []
        total_targ = []
        for i, (data, target, video_ids) in enumerate(dl):
            if True:
                if i == args.stop_after:
                    break
                # remove empty data, likely from preprocessing
                clean_data = []
                clean_targets = []
                for idx, element in enumerate(data):
                    if len(element) == 0:
                        logger.info(f'Empty data @ {video_ids[idx]}')
                        my_db = db.SponsorDB(MY_DB_PATH, no_setup=True)
                        table = 'subtitles' if utils.config['subtitletype'] == 'manual' else 'generated_subtitles'
                        my_db.delete_subtitles_by_videoid(table, video_ids[idx])
                        logger.info(f'Empty data @ {video_ids[idx]} deleted')
                    else:
                        clean_data.append(element)
                        clean_targets.append(target[idx])
                pbar.update()

                output = model(clean_data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(clean_targets, 0), args.cuda), requires_grad=False)
                loss = model.criterion(output, targets_var)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), clean_targets)
                total_loss += loss.item()

                if utils.config['type'] == 'classification':
                    output_seg = convert_class_to_seg(output_seg)
                    target_seg = convert_class_to_seg(target_seg)
                total_out.extend(output_seg)
                total_targ.extend(target_seg)

        avg_loss = total_loss / len(dl)
        epoch_windiff = windowdiff(total_targ, total_out, k=3, boundary=1)
        epoch_pk = pk(total_targ, total_out, boundary=1)
        logger.info(
            f'Validating Epoch: {epoch + 1}, accuracy: {preds_stats.get_accuracy():.4}, Pk: {epoch_pk:.4},' +
            f'Windiff: {epoch_windiff:.4}, F1: {preds_stats.get_f1():.6} ,Loss: {avg_loss:.6}')
        logger.info(f'TN: {preds_stats.tn} FN: {preds_stats.fn} FP: {preds_stats.fp} TP {preds_stats.tp}')
        preds_stats.reset()

        return epoch_pk, epoch_windiff


def get_params_from_log(log_path: str) -> Experiment_parameters:
    with open(log_path, 'r') as f:
        lines = f.readlines()
    lr = float(re.search("'lr': (\d.\d+)", lines[0]).group(1))
    seed = int(re.search("'seed': (\d+)", lines[0]).group(1))
    subtitle_type = re.search("'subtitletype': '(\w+)'", lines[0]).group(1) if re.search("'subtitletype': '(\w+)'", lines[0]) is not None else 'manual'
    hidden = int(re.search("'hidden_size': (\d+)", lines[0]).group(1))
    num_layers = int(re.search("'num_layers': (\d+)", lines[0]).group(1))
    datalen = int(re.search("'datalen': (\d+)", lines[0]).group(1))
    max_segment_number = int(re.search("'max_segment_number': (\d+)", lines[0]).group(1))
    type = re.search("'type': '(\w+)'", lines[0]).group(1) if re.search("'subtitletype': '(\w+)'", lines[0]) is not None else 'classification'
    new_params = Experiment_parameters(hidden=hidden,
                                       layers=num_layers,
                                       subtitle_type=subtitle_type,
                                       seed=seed,
                                       lr=lr,
                                       datalen=datalen,
                                       max_segment_number=max_segment_number,
                                       type = type)
    return new_params


def main(args: Namespace):
    experiment_parameters = get_params_from_log(args.log)
    config_file = './config/config.json'
    checkpoint_dir = os.path.split(args.load_from)[0]
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(checkpoint_dir, 'eval.log'), level=logging.DEBUG)

    utils.read_config_file(config_file)
    #utils.config.update(experiment_parameters.__dict__)
    utils.config['type'] = experiment_parameters.type
    utils.config['subtitletype'] = experiment_parameters.subtitle_type
    logger.debug('Running with config %s', utils.config)

    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
    model = maybe_cuda(model)

    my_db = db.SponsorDB(MY_DB_PATH)
    if experiment_parameters.subtitle_type == 'manual':
        unique_videos = my_db.get_unique_video_ids_from_subtitles()
    elif experiment_parameters.subtitle_type == 'generated':
        unique_videos = my_db.get_unique_video_ids_from_generated_subtitles()
    else:
        raise ValueError(
            f'{experiment_parameters.subtitle_type} is not a recognized subtitle type. Try manual or generated')
    random.Random(experiment_parameters.seed).shuffle(unique_videos)
    unique_videos = unique_videos[:experiment_parameters.datalen]

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    subtitle_type = 'subtitles_db' if experiment_parameters.subtitle_type == 'manual' else 'generated_subtitles_db'

    test_dataset = SubtitlesDataset(MY_DB_PATH, word2vec, unique_videos[int(len(unique_videos) * 0.9):],
                                    max_segments=experiment_parameters.max_segment_number, subtitle_type=subtitle_type,
                                    target_type=experiment_parameters.type)

    test_dl = DataLoader(test_dataset, batch_size=5, collate_fn=collate_fn, shuffle=False,
                         num_workers=0)

    print(model)

    test(model, args, -1, test_dl, logger)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load for testing')
    parser.add_argument('--log', help='Location of a log file containing parameters')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)

    main(parser.parse_args())
