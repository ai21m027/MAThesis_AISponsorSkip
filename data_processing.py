import pandas as pd
import tqdm
import database as SQL
import logging
import data_download_parallel

pd.set_option('display.max_columns', None)

MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'

def plausibility_check_and_clean(video_id,subtitle_type='manual'):
    my_db = SQL.SponsorDB(MY_DB_PATH)
    if subtitle_type == 'manual':
        subtitles_list = my_db.get_subtitles_by_videoid(video_id)
    elif subtitle_type == 'generated':
        subtitles_list = my_db.get_generated_subtitles_by_videoid(video_id)
    if len(subtitles_list) < 20:
        logging.error(f'There are not enough subtitles for video {video_id}, only {len(subtitles_list)} are available')
        if subtitle_type == 'manual':
            my_db.delete_subtitles_by_videoid(video_id)
        elif subtitle_type == 'generated':
            my_db.delete_generated_subtitles_by_videoid(video_id)
        return 1
    return 0

def add_sponsor_info_to_subtitle(video_id: str,subtitle_type='manual'):
    my_db = SQL.SponsorDB(MY_DB_PATH)
    sponsor_info_list = my_db.get_sponsor_info_by_video_id(video_id)
    subtitles_list = my_db.get_subtitles_by_videoid(video_id)
    if len(subtitles_list) < 20:
        logging.error(f'There are not enough subtitles for video {video_id}')
        return 1
    for sponsor_info in sponsor_info_list:
        sponsor_start = sponsor_info[2]
        sponsor_end = sponsor_info[3]
        if sponsor_start == 0.0 and sponsor_end == 0.0:
            logging.error(f'Sponsor info for video {video_id} starts and ends with 0.0')
            if len(sponsor_info) == 1:
                my_db.delete_sponsor_info_by_video_id(video_id)
                my_db.delete_subtitles_by_videoid(video_id)
        #print(sponsor_start, sponsor_end)
        for idx,subtitle in enumerate(subtitles_list):
            segment_start = subtitle[3]
            segment_duration = subtitle[4]
            text = subtitle[1]
            #print(segment_start, segment_duration)
            # Plus 1 second is for margin of error
            if segment_start > sponsor_start and segment_start + segment_duration < sponsor_end+1:
                #print(text)
                segment = SQL.SubtitleSegment(video_id=video_id,text=text,start_time=segment_start,duration=segment_duration,is_sponsor=True)
                if subtitle_type == 'manual':
                    my_db.update_subtitle(segment)
                elif subtitle_type == 'generated':
                    my_db.update_generated_subtitle(segment)
                else:
                    raise ValueError(f'Unknown subtitle_type {subtitle_type}')
    return 0

if __name__ == '__main__':
    logging.basicConfig(filename='log/processing_subtitles.log', encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    my_db = SQL.SponsorDB(MY_DB_PATH)

    unique_videos = my_db.get_unique_video_ids_from_generated_subtitles()
    print(len(unique_videos))

    short_count = 0
    print('Checking short subtitles...')
    for video_id in tqdm.tqdm(unique_videos,total=len(unique_videos)):
        short_count+=plausibility_check_and_clean(video_id[0],subtitle_type='generated')
    print(f'There are {short_count} too short subtitles of {len(unique_videos)} videos')
    for video_id in tqdm.tqdm(unique_videos,total=len(unique_videos)):
        short_count+= add_sponsor_info_to_subtitle(video_id[0],subtitle_type='generated')
    print(f'There are {short_count} too short subtitles of {len(unique_videos)} videos')
    exit()

