import pandas as pd
import database as SQL
pd.set_option('display.max_columns', None)

MY_DB_PATH = 'data/SQLite_YTSP_subtitles.db'

if __name__=='__main__':
    my_db = SQL.SponsorDB(MY_DB_PATH)
    data = pd.DataFrame(my_db.read_all_subtitle_info(),
                        columns=['video_id', 'text','issponsor', 'startTime', 'duration'])
    unique_videos = data['video_id'].unique()
    print(unique_videos)
    print(len(unique_videos))
    #print(data)