import gensim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import timedelta
from argparse import ArgumentParser, Namespace
import database as db
import utils
from subtitlesloader import SubtitlesDataset, collate_fn
from youtube_transcript_api import YouTubeTranscriptApi
import logging

from torch.autograd import Variable
from utils import maybe_cuda

DATABASE_PATH = 'data/SQLite_YTSP_subtitles.db'

logging.basicConfig(filename='log/execute.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def youtube_download(video_id: str, generated: bool = False):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        if generated:
            english_transcript = transcript_list.find_generated_transcript(['en', 'en-US'])
        else:
            english_transcript = transcript_list.find_manually_created_transcript(['en', 'en-US'])
        transcript = english_transcript.fetch()
    except:
        return None
    return transcript


def download_data(video_id: str) -> list:
    """
    Downloads either the manual or generated english subtitles for a given YouTube video id
    :param video_id: id of the video to be downloaded
    :return: List subtitles segments
    """
    segments = []
    transcript = youtube_download(video_id, generated=False)
    if transcript is None:  # if subtitles cannot be downloaded/don't exist skip
        transcript = youtube_download(video_id, generated=True)
        if transcript is None:
            logging.info(f'No subtitles for {video_id}')
            print(f'No subtitles for {video_id}')
            exit(-1)
    if len(transcript) < 20:
        logging.info(f'Subtitles too short for {video_id}')
        print(f'Subtitles too short for {video_id}')
        exit(-1)
    for line_dict in transcript:
        new_subtitle_segment = (video_id, line_dict['text'], 0, line_dict['start'], line_dict['duration'])
        segments.append(new_subtitle_segment)
    logging.info(f'Video: {video_id} downloaded')
    return segments


def evaluate(model: torch.nn.Module, dataset: DataLoader) -> tuple:
    """
    Returns the results of model processing the data from the Dataloader
    :param model: Model used for processing
    :param dataset: Dataloder which contains the data to be processed
    :return: list of tuples representing the class probabilities, corresponding targets if data is from database
    """
    model.eval()
    output_softmax = None
    clean_targets = None
    for i, (data, target, video_ids) in enumerate(dataset):
        clean_data = []
        clean_targets = []

        for idx, element in enumerate(data):
            if len(element) == 0:
                raise IndexError(f'{video_ids[idx]} cannot be processed for evaluation')
            else:
                clean_data.append(element)
                clean_targets.append(target[idx])

        output = model(clean_data)
        output_softmax = F.softmax(output, 1)

    return output_softmax, clean_targets


def output_to_timestamps(output: list, video_id: str) -> list:
    """
    Converts a list of classes to timestamps and text of the sponsor segments.
    :param output: List of classes for the segments
    :param video_id: id of the video to be processed
    :return: A list of sponsor segments, conatining timestamps
    """
    prev_target = 0
    in_segment = False
    start_time = 0
    end_time = 0
    my_db = db.SponsorDB(DATABASE_PATH)
    subtitles = my_db.get_subtitles_by_videoid(video_id)
    segments_list = []
    for idx, t in enumerate(output):
        target = int(t)
        if prev_target != target and target == 1:
            in_segment = True
            start_time = subtitles[idx][3]
            end_time = start_time
        elif prev_target != target and target == 0:
            segments_list.append((str(timedelta(seconds=start_time)), str(timedelta(seconds=end_time))))
            in_segment = False
        elif idx == len(output) - 1 and in_segment:
            duration = subtitles[idx][4]
            end_time += duration
            segments_list.append((str(timedelta(seconds=start_time)), str(timedelta(seconds=end_time))))
        if in_segment:
            duration = subtitles[idx][4]
            end_time += duration

        prev_target = target

    return segments_list


def output_to_timestamps_downloaded(output: list, subtitles: list) -> tuple:
    """
    Converts a list of classes to timestamps and text of the sponsor segments.
    :param output: List of classes for the segments
    :param subtitles: The subtitle segments to be proccessed
    :return: A tuple of sponsor segment timestamps and text
    """
    prev_target = 0
    in_segment = False
    start_time = 0
    segments_list = []
    text_list = []
    for idx, t in enumerate(output):
        target = int(t)
        if prev_target != target and target == 1 and not in_segment:
            in_segment = True
            start_time = subtitles[idx][3]
        elif prev_target != target and target == 0 and in_segment:
            end_time = subtitles[idx][3]
            segments_list.append((str(timedelta(seconds=start_time)), str(timedelta(seconds=end_time))))
            text_list.append('----------------------------------------------------------------')
            in_segment = False
        elif idx == len(output) - 1 and in_segment:
            end_time = subtitles[idx][3]
            duration = subtitles[idx][4]
            end_time += duration
            text_list.append(subtitles[idx][1])
            text_list.append('----------------------------------------------------------------')
            segments_list.append((str(timedelta(seconds=start_time)), str(timedelta(seconds=end_time))))
            in_segment = False
        if in_segment:
            text_list.append(subtitles[idx][1])
        prev_target = target
    return segments_list, text_list


def main(args: Namespace):
    test_video_id = args.videoid
    config_file = './config/config.json'
    utils.read_config_file(config_file)
    model_path = args.model
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    device = torch.device('cpu')
    model.to(device)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    if args.in_db:
        eval_ds = SubtitlesDataset(DATABASE_PATH, word2vec, [test_video_id], subtitle_type='subtitles_db')
        eval_dl = DataLoader(eval_ds, batch_size=1, collate_fn=collate_fn, shuffle=False,
                             num_workers=0)
    else:
        subtitle_segments = download_data(test_video_id)
        eval_ds = SubtitlesDataset(DATABASE_PATH, word2vec, [test_video_id], subtitle_type='execute',
                                   execute_subtitles=subtitle_segments)
        eval_dl = DataLoader(eval_ds, batch_size=1, collate_fn=collate_fn, shuffle=False,
                             num_workers=0)

    output, targets = evaluate(model, eval_dl, args)
    print(torch.argmax(output, dim=1))

    if args.in_db:
        print(targets)
        segments, text = output_to_timestamps(torch.argmax(output, dim=1), test_video_id)

        for t in text:
            print(t)
        print(list(segments))
        segments, text = output_to_timestamps(targets[0], test_video_id)
        print(segments)

    else:
        segments, text = output_to_timestamps_downloaded(torch.argmax(output, dim=1), subtitle_segments)

        for t in text:
            print(t)
        print(list(segments))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--in_db', help='Is video already in DB', action='store_true')
    parser.add_argument('--videoid', help='ID of input video', default='H6u0VBqNBQ8')
    parser.add_argument('--model', help='Path to the model')
    main(parser.parse_args())
