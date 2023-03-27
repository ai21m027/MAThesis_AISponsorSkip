import pandas as pd
import tqdm
import database as SQL
import logging
from argparse import ArgumentParser

pd.set_option('display.max_columns', None)

MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'


def plausibility_check_and_clean(video_id: str, subtitle_type: str = 'manual') -> int:
    """
    Checks a video id in the database for potential errors and deletes the entry from the database.
    :param video_id: video id to be checked in the database
    :param subtitle_type: Either manual or generated depending on type of subtitle to be checkec
    :return: Number of errors found in the subtitles
    """
    my_db = SQL.SponsorDB(MY_DB_PATH, no_setup=True)
    if subtitle_type == 'manual':
        subtitles_list = my_db.get_subtitles_by_videoid(video_id)
    elif subtitle_type == 'generated':
        subtitles_list = my_db.get_generated_subtitles_by_videoid(video_id)
    else:
        raise ValueError(f'Unknown subtitle_type {subtitle_type}')
    if len(subtitles_list) < 20:
        logging.error(f'There are not enough subtitles for video {video_id}, only {len(subtitles_list)} are available')
        if subtitle_type == 'manual':
            my_db.delete_subtitles_by_videoid(video_id)
        elif subtitle_type == 'generated':
            my_db.delete_generated_subtitles_by_videoid(video_id)
        return 1
    return 0


def add_sponsor_info_to_subtitle(video_id: str, subtitle_type: str = 'manual') -> None:
    """
    Adds the information in the sponsor_info table to the issponsor column  in either the
    subtitles or generated_subtitles table.
    :param video_id: video id of the subtitles to be processed
    :param subtitle_type: Either manual or generated depending on type of subtitle
    """
    my_db = SQL.SponsorDB(MY_DB_PATH)
    sponsor_info_list = my_db.get_sponsor_info_by_video_id(video_id)
    if subtitle_type == 'manual':
        subtitles_list = my_db.get_subtitles_by_videoid(video_id)
    elif subtitle_type == 'generated':
        subtitles_list = my_db.get_generated_subtitles_by_videoid(video_id)
    for sponsor_info in sponsor_info_list:
        sponsor_start = sponsor_info[2]
        sponsor_end = sponsor_info[3]
        if sponsor_start == 0.0 and sponsor_end == 0.0:
            logging.error(f'Sponsor info for video {video_id} starts and ends with 0.0')
            if len(sponsor_info) == 1:
                my_db.delete_sponsor_info_by_video_id(video_id)
                my_db.delete_subtitles_by_videoid(video_id)
        for idx, subtitle in enumerate(subtitles_list):
            segment_start = subtitle[3]
            segment_duration = subtitle[4]
            text = subtitle[1]
            # Plus 1 second is for margin of error
            if segment_start > sponsor_start and segment_start + segment_duration < sponsor_end + 1:
                segment = SQL.SubtitleSegment(video_id=video_id, text=text, start_time=segment_start,
                                              duration=segment_duration, is_sponsor=True)
                if subtitle_type == 'manual':
                    my_db.update_subtitle(segment)
                elif subtitle_type == 'generated':
                    my_db.update_generated_subtitle(segment)
                else:
                    raise ValueError(f'Unknown subtitle_type {subtitle_type}')


def main(args) -> None:
    logging.basicConfig(filename='log/processing_subtitles.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    db = SQL.SponsorDB(MY_DB_PATH)
    mode = args.mode
    if mode == 'generated':
        unique_videos = db.get_unique_video_ids_from_generated_subtitles()
        print(len(unique_videos))
    elif mode == 'manual':
        unique_videos = db.get_unique_video_ids_from_subtitles()
        print(len(unique_videos))
    else:
        raise ValueError(f'Unknown mode {mode}. Use manual or generated.')

    short_count = 0
    print('Checking short subtitles...')
    for videoid in tqdm.tqdm(unique_videos, total=len(unique_videos)):
        short_count += plausibility_check_and_clean(videoid[0], subtitle_type=mode)
    print(f'There were {short_count} too short subtitles of {len(unique_videos)} videos that have been removed')
    logging.info(f'There were {short_count} too short subtitles of {len(unique_videos)} videos that have been removed')
    for videoid in tqdm.tqdm(unique_videos, total=len(unique_videos)):
        add_sponsor_info_to_subtitle(videoid[0], subtitle_type=mode)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', help='Type of subtitle to download. Either manual or generated', type=str,
                        default='manual')
    main(parser.parse_args())
