import gensim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import timedelta
from argparse import ArgumentParser
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

def youtube_download(video_id: str,generated:bool=False):
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


def download_data(video_id):
    segments = []
    transcript = youtube_download(video_id,generated=False)
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
        new_subtitle_segment = (video_id,line_dict['text'],0,line_dict['start'],line_dict['duration'])
        #new_subtitle_segment = db.SubtitleSegment(video_id=video_id, text=line_dict['text'],is_sponsor=False, start_time=line_dict['start'],duration=line_dict['duration'])
        segments.append(new_subtitle_segment)
    logging.info(f'Video: {video_id} downloaded')
    return segments


def evaluate(model, dataset,args):
    model.eval()

    total_loss = float(0)
    for i, (data, target, video_ids) in enumerate(dataset):
        # remove empty data, likely from preprocessing
        clean_data = []
        clean_targets = []

        for idx, element in enumerate(data):
            if len(element) == 0:
                # logger.info(f'Empty data @ {video_ids[idx]}')
                pass
            else:
                clean_data.append(element)
                clean_targets.append(target[idx])
        #clean_data = Variable(maybe_cuda(torch.cat(clean_data, 0), args.cuda), requires_grad=False)
        output = model(clean_data)
        output_softmax = F.softmax(output, 1)

    return output_softmax, clean_targets


def output_to_timestamps(output, video_id):
    prev_target = 0
    in_segment = False
    start_time = 0
    end_time = 0
    my_db = db.SponsorDB(DATABASE_PATH)
    subtitles = my_db.get_subtitles_by_videoid(video_id)
    segments_list =[]
    for idx, t in enumerate(output):
        target = int(t)
        if prev_target != target and target == 1:
            in_segment = True
            start_time = subtitles[idx][3]
            end_time = start_time
        elif (prev_target != target and target == 0) :
            segments_list.append((str(timedelta(seconds=start_time)), str(timedelta(seconds=end_time))))
            in_segment = False
        elif idx == len(output) - 1 and in_segment==True:
            duration = subtitles[idx][4]
            end_time += duration
            segments_list.append((str(timedelta(seconds=start_time)), str(timedelta(seconds=end_time))))
        if in_segment:
            duration = subtitles[idx][4]
            end_time += duration

        prev_target = target

    return segments_list



def output_to_timestamps_downloaded(output, subtitles):
    prev_target = 0
    in_segment = False
    start_time = 0
    end_time = 0
    my_db = db.SponsorDB(DATABASE_PATH)
    segments_list =[]
    text_list = []
    for idx, t in enumerate(output):
        target = int(t)
        if prev_target != target and target == 1 and not in_segment:
            in_segment = True
            start_time = subtitles[idx][3]
        elif (prev_target != target and target == 0 and in_segment) :
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
            in_segment =False
        if in_segment:
            text_list.append(subtitles[idx][1])

        prev_target = target

    return segments_list,text_list

def main(args):
    test_video_id = 'H6u0VBqNBQ8'
    # test_video_id = 'WD9jKHNx-E4'
    test_video_id = args.videoid
    config_file = './config/config.json'
    utils.read_config_file(config_file)
    model_path = r'D:\Root_Philipp_Laptop\FH_AI\MA_Arbeit\Experiments\ModelSize_2nd_finished\256_2_1234\model009.t7'
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    device = torch.device('cpu')
    model.to(device)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    if args.in_db:
        eval_ds = SubtitlesDataset(DATABASE_PATH, word2vec, [test_video_id],mode='subtitles_db')
        eval_dl = DataLoader(eval_ds, batch_size=1, collate_fn=collate_fn, shuffle=False,
                             num_workers=0)
    else:
        subtitle_segments = download_data(test_video_id)
        #print(subtitle_segments)
        eval_ds = SubtitlesDataset(DATABASE_PATH, word2vec, [test_video_id],mode='execute',excute_subtitles=subtitle_segments)
        eval_dl = DataLoader(eval_ds, batch_size=1, collate_fn=collate_fn, shuffle=False,
                             num_workers=0)


    output, targets = evaluate(model, eval_dl,args)
    print(torch.argmax(output, dim=1))

    if args.in_db:
        print(targets)
        segments,text = output_to_timestamps(torch.argmax(output, dim=1),test_video_id)

        for t in text:
            print(t)
        print(list(segments))
        segments,text=output_to_timestamps(targets[0],test_video_id)
        print(segments)

    else:
        segments,text = output_to_timestamps_downloaded(torch.argmax(output, dim=1),subtitle_segments)

        for t in text:
            print(t)
        print(list(segments))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--in_db', help='Is video already in DB', action='store_true')
    parser.add_argument('--videoid', help='ID of input video', default='H6u0VBqNBQ8')
    main(parser.parse_args())
