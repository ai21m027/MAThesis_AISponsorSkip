import pandas as pd
import database as SQL
pd.set_option('display.max_columns', None)

MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'

def add_sponsor_info_to_subtitle(video_id:str):
    my_db = SQL.SponsorDB(MY_DB_PATH)
    sponsor_info_list = my_db.get_sponsor_info_by_video_id(video_id)
    for sponsor_info in sponsor_info_list:
        print(sponsor_info)



if __name__=='__main__':
    my_db = SQL.SponsorDB(MY_DB_PATH)

    unique_videos = my_db.get_unique_video_ids_from_subtitles()
    print(len(unique_videos))

    #add_sponsor_info_to_subtitle(unique_videos[0][0])
    #exit()
    for video_id in unique_videos:
        add_sponsor_info_to_subtitle(video_id[0])
    #print(data)