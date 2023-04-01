import logging
import os
import random
from argparse import ArgumentParser, Namespace
from logging import Logger

import gensim
import numpy as np
import torch
import torch.nn.functional as F
from pathlib2 import Path
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.metrics import windowdiff, pk
import accuracy
import database as db
import models.dualLSTMmodel as Dlstm
import utils
from subtitlesloader import SubtitlesDataset, collate_fn
from utils import maybe_cuda

torch.multiprocessing.set_sharing_strategy('file_system')
preds_stats = utils.predictions_analysis()
MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'


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


def train(model: torch.nn.Module, args: Namespace, epoch: int, dl: DataLoader, logger: Logger,
          optimizer: torch.optim.Optimizer) -> None:
    """
    Trains a ML model on the data provided by the dl Dataloader for one epoch
    :param model: ML Model to be trained
    :param args: general congfig args
    :param epoch: current training epoch, used for logging
    :param dl: Dataloader for training data
    :param logger:
    :param optimizer: ML Optimizer for training
    """
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dl)) as pbar:
        for i, (data, target, video_ids) in enumerate(dl):
            if i == args.stop_after:
                break
            # remove empty data, from preprocessing
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
            model.zero_grad()
            output = model(clean_data)
            # unused because does not work for training
            # output_softmax = F.softmax(output, 1)
            target_var = Variable(maybe_cuda(torch.cat(clean_targets, 0), args.cuda), requires_grad=False)
            loss = model.criterion(output, target_var)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            pbar.set_description('Training, loss={:.4}'.format(loss.item()))

    avg_loss = total_loss / len(dl)
    logger.info('Training Epoch: {}, Loss: {:.6}.'.format(epoch + 1, avg_loss))


def validate(model: torch.nn.Module, args: Namespace, epoch: int, dl: DataLoader, logger: Logger) -> (
        float, float):
    """
    Validates a ML model on the given dataloader
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


def main(args: Namespace) -> None:
    """
    :param args:
    :return:
    """
    config_file = './config/config.json'
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    logger = utils.setup_logger(__name__, os.path.join(checkpoint_dir, 'train.log'), level=logging.DEBUG)

    utils.read_config_file(config_file)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)
    if args.load_from is None:
        model = Dlstm.create(hidden_size=args.hidden_size, num_layers=args.num_layers)
        model = maybe_cuda(model)
    else:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)
        model = maybe_cuda(model)

    my_db = db.SponsorDB(MY_DB_PATH)
    if args.subtitletype == 'manual':
        unique_videos = my_db.get_unique_video_ids_from_subtitles()
    elif args.subtitletype == 'generated':
        unique_videos = my_db.get_unique_video_ids_from_generated_subtitles()
    else:
        raise ValueError(f'{args.subtitle_type} is not a recognized subtitle type. Try manual or generated')
    random.Random(args.seed).shuffle(unique_videos)
    unique_videos = unique_videos[:args.datalen]

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    subtitle_type = 'subtitles_db' if args.subtitletype == 'manual' else 'generated_subtitles_db'

    train_dataset = SubtitlesDataset(MY_DB_PATH, word2vec, unique_videos[:int(len(unique_videos) * 0.8)],
                                     max_segments=args.max_segment_number, subtitle_type=subtitle_type,
                                     target_type=args.type)
    dev_dataset = SubtitlesDataset(MY_DB_PATH, word2vec,
                                   unique_videos[int(len(unique_videos) * 0.8):int(len(unique_videos) * 0.9)],
                                   max_segments=args.max_segment_number, subtitle_type=subtitle_type,
                                   target_type=args.type)
    test_dataset = SubtitlesDataset(MY_DB_PATH, word2vec, unique_videos[int(len(unique_videos) * 0.9):],
                                    max_segments=args.max_segment_number, subtitle_type=subtitle_type,
                                    target_type=args.type)

    train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=False,
                          num_workers=args.num_workers)
    dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                        num_workers=args.num_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                         num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)
    for j in range(args.epochs):
        train(model, args, j, train_dl, logger, optimizer)
        with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
            torch.save(model, f)
        validate(model, args, j, dev_dl, logger)
    test(model, args, utils.config['epochs'], test_dl, logger)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue',
                        default=None)
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--type', help='Type of processing. Either classification or segmentation',
                        default='classification')
    parser.add_argument('--subtitletype', help='Type of subtitle to be processed. Either manual or generated',
                        default='manual')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--seed', help='Seed for training selection', type=int, default=42)
    parser.add_argument('--datalen', help='Length of training data to use', type=int, default=-1)
    parser.add_argument('--num_layers', help='Number of layers per LSTM', type=int, default=2)
    parser.add_argument('--hidden_size', help='Size of hidden layers', type=int, default=256)
    parser.add_argument('--max_segment_number', help='Maximum number of segments in a video', type=int, default=300)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=1e-3)

    main(parser.parse_args())
