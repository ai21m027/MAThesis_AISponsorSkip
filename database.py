import sqlite3
from typing import NamedTuple
import pandas as pd


class SponsorInfo(NamedTuple):
    video_id: str
    start_time: float
    end_time: float
    upvotes: int
    downvotes: int


class SubtitleSegment(NamedTuple):
    video_id: str
    text: str
    is_sponsor: bool
    start_time: float
    duration: float


class SponsorDB():
    _cursor = None
    _db_name = None
    _sqliteConnection = None
    _nr_instances = 0

    def __init__(self, db_name: str):
        if SponsorDB._nr_instances == 0:
            SponsorDB._db_name = db_name
            self.init_database()
            print(f'Connection to {SponsorDB._db_name} opened')
        elif db_name != SponsorDB._db_name:
            raise ConnectionAbortedError(f'SponsorDB already exists with different Database: {SponsorDB._db_name}')
        SponsorDB._nr_instances += 1

    def __del__(self):
        if SponsorDB._nr_instances <= 1:
            SponsorDB._cursor.close()
            SponsorDB._sqliteConnection.close()
            print(f'Connection to {SponsorDB._db_name} closed')
        SponsorDB._nr_instances -= 1

    def init_database(self):
        # if database does not exist sqlite3.connect creates the database
        SponsorDB._sqliteConnection = sqlite3.connect(self._db_name)
        SponsorDB._cursor = self._sqliteConnection.cursor()
        # creating sponsor_info table
        q_create_table = '''CREATE TABLE IF NOT EXISTS sponsor_info (
                            uid INTEGER PRIMARY KEY AUTOINCREMENT,
                            videoid TEXT NOT NULL,
                            startTime FLOAT NOT NULL,
                            endTime FLOAT NOT NULL,
                            upvotes INTEGER NOT NULL,
                            downvotes INTEGER NOT NULL);'''
        self._cursor.execute(q_create_table)
        self._sqliteConnection.commit()
        print('Table sponsor_info ok')
        q_create_index = '''CREATE UNIQUE INDEX IF NOT EXISTS vid_start_idx ON sponsor_info (videoid,startTime)'''
        self._cursor.execute(q_create_index)
        self._sqliteConnection.commit()
        print('vid_start_idx ok')
        # creating subtitles table
        q_create_table = '''CREATE TABLE IF NOT EXISTS subtitles (
                            uid INTEGER PRIMARY KEY AUTOINCREMENT,
                            videoid TEXT NOT NULL,
                            text TEXT NOT NULL,
                            issponsor BOOLEAN DEFAULT FALSE,
                            startTime FLOAT NOT NULL,
                            duration FLOAT NOT NULL);'''
        self._cursor.execute(q_create_table)
        self._sqliteConnection.commit()
        print('Table subtitles ok')

    def store_sponsor_info(self, s_info: SponsorInfo):
        try:

            q_write_sponsor_info = '''INSERT INTO sponsor_info
                                    (videoid,startTime,endTime,upvotes,downvotes)
                                    VALUES (?,?,?,?,?)
                                    '''
            self._cursor.execute(q_write_sponsor_info, (s_info.video_id, s_info.start_time,
                                                        s_info.end_time, s_info.upvotes, s_info.downvotes))
            self._sqliteConnection.commit()
        except sqlite3.IntegrityError as error:
            test = error
            print("Error while connecting to sqlite", error, s_info.video_id, s_info.start_time)

    def read_all_sponsor_info(self):
        q_read_sponsor_info = '''SELECT * FROM sponsor_info
                                '''
        self._cursor.execute(q_read_sponsor_info)
        return self._cursor.fetchall()

    def read_subtitles_by_videoid(self,video_id:str)->list:
        q_read_subtitles = '''SELECT * FROM subtitles WHERE videoid = ?'''
        self._cursor.execute(q_read_subtitles,(video_id,))
        return self._cursor.fetchall()


if __name__ == '__main__':
    my_sponsor_db = SponsorDB('data/SQLite_YTSP_subtitles.db')
    my_spi = SponsorInfo(video_id='test123', start_time=62.4, end_time=4003.7, upvotes=23,
                         downvotes=7)
    # my_sponsor_db.store_sponsor_info(my_spi)
    test = pd.DataFrame(my_sponsor_db.read_all_sponsor_info(),columns=['uid','video_id','start_time','end_time','upvotes','downvotes'])
    test = test.sort_values('upvotes',ascending=False)
    test = test.reset_index(drop=True)
    print(test)
