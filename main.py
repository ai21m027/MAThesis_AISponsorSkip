import logging
import os
import random
from argparse import ArgumentParser

import gensim
import numpy as np
import torch
import torch.nn.functional as F
from pathlib2 import Path
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# from wiki_loader import WikipediaDataSet
import accuracy
import database as db
import models.dualLSTMmodel as M
import utils
from subtitlesloader import SubtitlesDataset, collate_fn
from utils import maybe_cuda

# from termcolor import colored

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


def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, video_ids) in enumerate(dataset):
            if i == args.stop_after:
                break

            pbar.update()
            model.zero_grad()
            output = model(data)
            target_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
            loss = model.criterion(output, target_var)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            # logger.debug('Batch %s - Train error %7.4f', i, loss.data[0])
            pbar.set_description('Training, loss={:.4}'.format(loss.item()))
            # except Exception as e:
            # logger.info('Exception "%s" in batch %s', e, i)
            # logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            # pass

    avg_loss = total_loss / len(dataset)
    logger.info('Training Epoch: {}, Loss: {:.6}.'.format(epoch + 1, avg_loss))
    # log_value('Training Loss', avg_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validating', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, video_ids) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)

                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), target)

            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.6} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            preds_stats.get_f1()))
        logger.info(f'TN: {preds_stats.tn} FN: {preds_stats.fn} FP: {preds_stats.fp} TP {preds_stats.tp}')
        preds_stats.reset()

        return epoch_pk, threshold


def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = accuracy.Accuracy()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                current_idx = 0

                for k, t in enumerate(target):
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)

                    output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    h = np.append(output, [1])
                    tt = np.append(t, [1])

                    acc.update(h, tt)

                    current_idx = to_idx

                    # acc.update(output_softmax.data.cpu().numpy(), target)

            #
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk,
                                                                                                          epoch_windiff,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk


def main(args):
    config_file = './config/config.json'
    checkpoint_dir = 'checkpoints'
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    logger = utils.setup_logger(__name__, os.path.join(checkpoint_dir, 'train.log'), level=logging.DEBUG)

    utils.read_config_file(config_file)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)
    model = M.create()
    model = maybe_cuda(model)

    my_db = db.SponsorDB(MY_DB_PATH)
    unique_videos = my_db.get_unique_video_ids_from_subtitles()
    random.Random(args.seed).shuffle(unique_videos)
    unique_videos = unique_videos[:args.datalen]

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    train_dataset = SubtitlesDataset(MY_DB_PATH, word2vec, unique_videos[:int(len(unique_videos) * 0.8)])
    dev_dataset = SubtitlesDataset(MY_DB_PATH, word2vec,
                                   unique_videos[int(len(unique_videos) * 0.8):int(len(unique_videos) * 0.9)])
    test_dataset = SubtitlesDataset(MY_DB_PATH, word2vec, unique_videos[int(len(unique_videos) * 0.9):])

    train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=False,
                          num_workers=args.num_workers)
    dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                        num_workers=args.num_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                         num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    best_val_pk = 1.0
    for j in range(args.epochs):
        train(model, args, j, train_dl, logger, optimizer)
        with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
            torch.save(model, f)
        val_pk, threshold = validate(model, args, j, dev_dl, logger)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    # parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    # parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--infer', help='inference_dir', type=str)
    parser.add_argument('--seed', help='Seed for training selection', type=int, default=42)
    parser.add_argument('--datalen', help='Length of training data to use', type=int, default=-1)

    main(parser.parse_args())
