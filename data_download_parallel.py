import logging

import pandas as pd
from rich.progress import track
from argparse import ArgumentParser

pd.set_option('display.max_columns', None)
from multiprocessing.pool import Pool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
import time
import database as SQL
import tqdm

MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'

logging.basicConfig(filename='log/subtitles_download.log', encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def youtube_download(video_id: str, generated: bool = False)->list[dict]:
    """
    Downloads either the manual or generated english transcript for a given video_id
    :param video_id: id of the video to download
    :param generated: true for generated subtitles, false for manual subtitles
    :return: A list containing a dictionary for each subtitle segment
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        if generated:
            english_transcript = transcript_list.find_generated_transcript(['en', 'en-US'])
        else:
            english_transcript = transcript_list.find_manually_created_transcript(['en', 'en-US'])
        transcript = english_transcript.fetch()
    except NoTranscriptFound:
        return None
    return transcript


def write_to_sponsor_info(input_data: pd.DataFrame) -> None:
    i_sponsor_db = SQL.SponsorDB(MY_DB_PATH)
    for idx, element in track(input_data.iterrows(), description='Writing data to sql'):
        if element.loc['category'] == 'sponsor':
            new_sponsor = SQL.SponsorInfo(
                video_id=element.loc['videoID'],
                start_time=element.loc['startTime'],
                end_time=element.loc['endTime'],
                upvotes=element.loc['votes'],
                downvotes=element.loc['incorrectVotes'],
            )
            i_sponsor_db.store_sponsor_info(new_sponsor)


def download_and_write_subtitles(video_id: str) -> None:
    """
    Downloads the manually generated english subtitles for the given video_id and writes it to the database.
    :param video_id: id of the video who's subtitles should be added to the database
    :return: None
    """
    i_sponsor_db = SQL.SponsorDB(MY_DB_PATH, no_setup=True)
    # check if subtitles are already in DB
    if i_sponsor_db.check_video_exists_in_subtitles(video_id):  # if subtitles already in DB skip
        logging.info(f'Video: {video_id} already in database.')
        return
    transcript = youtube_download(video_id)
    if transcript is None:  # if subtitles cannot be downloaded/don't exist skip
        logging.info(f'No subtitles for {video_id}')
        return
    if len(transcript) < 20:
        logging.info(f'Subtitles too short for {video_id}')
        return
    for line_dict in transcript:
        new_subtitle_segment = SQL.SubtitleSegment(video_id=video_id, text=line_dict['text'],
                                                   is_sponsor=False, start_time=line_dict['start'],
                                                   duration=line_dict['duration'])
        i_sponsor_db.store_subtitle(new_subtitle_segment)
    logging.info(f'Video: {video_id} added to database')
    return


def download_and_write_generated_subtitles(video_id: str) -> None:
    i_sponsor_db = SQL.SponsorDB(MY_DB_PATH, no_setup=True)
    # check if subtitles are already in DB
    if i_sponsor_db.check_video_exists_in_generated_subtitles(video_id):  # if subtitles already in DB skip
        logging.info(f'Video: {video_id} already in database.')
        return
    transcript = youtube_download(video_id, generated=True)
    if transcript is None:  # if subtitles cannot be downloaded/don't exist skip
        logging.info(f'No subtitles for {video_id}')
        return
    if len(transcript) < 20:
        logging.info(f'Subtitles too short for {video_id}')
        return
    for line_dict in transcript:
        new_subtitle_segment = SQL.SubtitleSegment(video_id=video_id, text=line_dict['text'],
                                                   is_sponsor=False, start_time=line_dict['start'],
                                                   duration=line_dict['duration'])
        i_sponsor_db.store_generated_subtitle(new_subtitle_segment)
    logging.info(f'Video: {video_id} added to database')
    return


def main(args):
    my_sponsor_db = SQL.SponsorDB(MY_DB_PATH)

    mode = args.mode
    logging.info(f'Mode: {mode}')
    data = pd.DataFrame(my_sponsor_db.get_all_sponsor_info(),
                        columns=['uid', 'video_id', 'start_time', 'end_time', 'upvotes', 'downvotes'])
    data = data.sort_values('upvotes', ascending=False)
    data = data.reset_index(drop=True)
    unique_video_ids = data['video_id'].unique()
    start = time.time()
    if mode == 'manual':
        with Pool() as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(download_and_write_subtitles, unique_video_ids),
                               total=len(unique_video_ids)):
                pass
    elif mode == 'generated':
        with Pool() as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(download_and_write_generated_subtitles, unique_video_ids),
                               total=len(unique_video_ids)):
                pass
    end = time.time()
    print(f"Download time: {end - start}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', help='Type of subtitle to download. Either manual or generated', type=str,
                        default='manual')
    main(parser.parse_args())
