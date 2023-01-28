import sqlite3
from typing import NamedTuple
from datetime import datetime,timedelta




class SponsorInfo(NamedTuple):
    video_id: str
    start_time: datetime
    end_time: datetime
    upvotes: int
    downvotes: int




class SponsorDB():
    _cursor = None
    _db_name = None
    _sqliteConnection=None
    _nr_instances=0
    def __init__(self,db_name:str):
        if self._nr_instances == 0:
            self._db_name = db_name
            self.init_database()
            print(f'Connection to {self._db_name} opened')
        self._nr_instances+=1
    def __del__(self):
        if self._nr_instances <=1:
            self._cursor.close()
            self._sqliteConnection.close()
            print(f'Connection to {self._db_name} closed')
        self._nr_instances-=1

    def init_database(self):
        # if database does not exist sqlite3.connect creates the database
        self._sqliteConnection = sqlite3.connect(self._db_name)
        self._cursor = self._sqliteConnection.cursor()

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
            print("Error while connecting to sqlite", error,s_info.video_id,s_info.start_time)



    def read_all_sponsor_info(self,s_storage:str):
        q_read_sponsor_info = '''SELECT * FROM sponsor_info
                                    WHERE videoid LIKE '%'
                                '''
        self._cursor.execute(q_read_sponsor_info)
        results = self._cursor.fetchall()
        print(results)
        return results


if __name__ == '__main__':
    my_sponsor_db = SponsorDB('data/SQLite_YTSP_subtitles.db')
    my_spi = SponsorInfo(video_id='test123', start_time=62.4, end_time=4003.7, upvotes=23,
                         downvotes=7)
    #my_sponsor_db.store_sponsor_info(my_spi)
