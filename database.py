import logging
import sqlite3
import time
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
    """
    Singelton Sponsor Database
    """
    _cursor = None
    _db_name = None
    _sqliteConnection = None
    _nr_instances = 0

    def __init__(self, db_name: str,no_setup:bool=False):
        self._db_name = db_name
        self.init_database(no_setup)
        logging.debug(f'Connection to {SponsorDB._db_name} opened')

    def __del__(self):
        if self._nr_instances <= 1:
            self._cursor.close()
            self._sqliteConnection.close()
            logging.debug(f'Connection to {SponsorDB._db_name} closed')
        self._nr_instances -= 1

    def init_database(self,no_setup:bool=False):
        # if database does not exist sqlite3.connect creates the database
        self._sqliteConnection = sqlite3.connect(self._db_name, timeout=30)
        self._cursor = self._sqliteConnection.cursor()
        # creating sponsor_info table
        if no_setup:
            return
        q_create_table = '''CREATE TABLE IF NOT EXISTS sponsor_info (
                            uid INTEGER PRIMARY KEY AUTOINCREMENT,
                            videoid TEXT NOT NULL,
                            startTime FLOAT NOT NULL,
                            endTime FLOAT NOT NULL,
                            upvotes INTEGER NOT NULL,
                            downvotes INTEGER NOT NULL);'''
        self._cursor.execute(q_create_table)
        self._sqliteConnection.commit()
        logging.debug('Table sponsor_info ok')
        q_create_index = '''CREATE UNIQUE INDEX IF NOT EXISTS vid_start_idx ON sponsor_info (videoid,startTime)'''
        self._cursor.execute(q_create_index)
        self._sqliteConnection.commit()
        logging.debug('vid_start_idx on sponsor_info ok')
        # creating subtitles table
        q_create_table = '''CREATE TABLE IF NOT EXISTS subtitles (
                            videoid TEXT NOT NULL,
                            text TEXT NOT NULL,
                            issponsor BOOLEAN DEFAULT FALSE,
                            startTime FLOAT NOT NULL,
                            duration FLOAT NOT NULL);'''
        self._cursor.execute(q_create_table)
        self._sqliteConnection.commit()
        logging.debug('Table subtitles ok')
        q_create_index = '''CREATE INDEX IF NOT EXISTS vidid_idx ON subtitles (videoid)'''
        self._cursor.execute(q_create_index)
        self._sqliteConnection.commit()
        logging.debug('vidid_idx on subtitles ok')

        # creating generated subtitles table
        q_create_table = '''CREATE TABLE IF NOT EXISTS generated_subtitles (
                            videoid TEXT NOT NULL,
                            text TEXT NOT NULL,
                            issponsor BOOLEAN DEFAULT FALSE,
                            startTime FLOAT NOT NULL,
                            duration FLOAT NOT NULL);'''
        self._cursor.execute(q_create_table)
        self._sqliteConnection.commit()
        logging.debug('Table generated_subtitles ok')
        q_create_index = '''CREATE INDEX IF NOT EXISTS vidid_idx ON generated_subtitles (videoid)'''
        self._cursor.execute(q_create_index)
        self._sqliteConnection.commit()
        logging.debug('vidid_idx on subtitles ok')

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

            logging.error("Error while connecting to sqlite", error, s_info.video_id, s_info.start_time)

    def store_subtitle(self, subtitle: SubtitleSegment, iteration: int = 0):
        if iteration >= 10:
            return
        try:

            q_write_subtitle = '''INSERT INTO subtitles
                                    (videoid,text,issponsor,startTime,duration)
                                    VALUES (?,?,?,?,?)
                                    '''
            self._cursor.execute(q_write_subtitle, (
                subtitle.video_id, subtitle.text, subtitle.is_sponsor, subtitle.start_time, subtitle.duration))
            self._sqliteConnection.commit()
        except sqlite3.IntegrityError as error:

            logging.error("Error while connecting to sqlite", error, subtitle.video_id, subtitle.start_time)

        except sqlite3.OperationalError as error:
            logging.error(f"Error while connecting to sqlite: {error}. Database locked. Iteration:{iteration}")
            time.sleep(1)
            self.store_subtitle(subtitle, iteration + 1)

    def store_generated_subtitle(self, subtitle: SubtitleSegment, iteration: int = 0):
        if iteration >= 10:
            return
        try:

            q_write_subtitle = '''INSERT INTO generated_subtitles
                                    (videoid,text,issponsor,startTime,duration)
                                    VALUES (?,?,?,?,?)
                                    '''
            self._cursor.execute(q_write_subtitle, (
                subtitle.video_id, subtitle.text, subtitle.is_sponsor, subtitle.start_time, subtitle.duration))
            self._sqliteConnection.commit()
        except sqlite3.IntegrityError as error:

            logging.error("Error while connecting to sqlite", error, subtitle.video_id, subtitle.start_time)

        except sqlite3.OperationalError as error:
            logging.error(f"Error while connecting to sqlite: {error}. Database locked. Iteration:{iteration}")
            time.sleep(1)
            self.store_subtitle(subtitle, iteration + 1)

    def update_subtitle(self, subtitle:SubtitleSegment):
        try:
            q_update_subtitle = '''UPDATE subtitles SET issponsor=?
            WHERE videoid=? AND startTime=? AND duration=?'''
            self._cursor.execute(q_update_subtitle, (subtitle.is_sponsor, subtitle.video_id,subtitle.start_time,subtitle.duration))
            self._sqliteConnection.commit()
        except sqlite3.IntegrityError as error:
            logging.error("Error while connecting to sqlite", error, subtitle.video_id, subtitle.start_time)
    def get_all_sponsor_info(self):
        q_read_sponsor_info = '''SELECT * FROM sponsor_info
                                '''
        self._cursor.execute(q_read_sponsor_info)
        return self._cursor.fetchall()

    def get_all_subtitle_info(self):
        q_read_subtitle_info = '''SELECT * FROM subtitles
                                '''
        self._cursor.execute(q_read_subtitle_info)
        return self._cursor.fetchall()

    def get_all_generated_subtitle_info(self):
        q_read_subtitle_info = '''SELECT * FROM generated_subtitles
                                '''
        self._cursor.execute(q_read_subtitle_info)
        return self._cursor.fetchall()

    def get_subtitles_by_videoid(self, video_id: str) -> list:
        q_read_subtitles = '''SELECT * FROM subtitles WHERE videoid = ? ORDER BY startTime ASC'''
        self._cursor.execute(q_read_subtitles, (video_id,))
        return self._cursor.fetchall()

    def get_generated_subtitles_by_videoid(self, video_id: str) -> list:
        q_read_subtitles = '''SELECT * FROM generated_subtitles WHERE videoid = ?'''
        self._cursor.execute(q_read_subtitles, (video_id,))
        return self._cursor.fetchall()

    def get_sponsor_info_by_video_id(self, video_id: str) -> list:
        q_read_sponsor_info = '''SELECT * FROM sponsor_info WHERE videoid =? ORDER BY startTime ASC'''
        self._cursor.execute(q_read_sponsor_info, (video_id,))
        return self._cursor.fetchall()

    def get_unique_video_ids_from_subtitles(self) -> list:
        q_read_subtitles = '''SELECT DISTINCT videoid FROM subtitles'''
        self._cursor.execute(q_read_subtitles)
        id_list = [element[0] for element in self._cursor.fetchall()]
        return id_list

    def get_unique_video_ids_from_generated_subtitles(self) -> list:
        q_read_subtitles = '''SELECT DISTINCT videoid FROM generated_subtitles'''
        self._cursor.execute(q_read_subtitles)
        return self._cursor.fetchall()

    def check_video_exists_in_generated_subtitles(self, video_id: str) -> bool:
        q_read_generated_subtitles = '''SELECT 1 FROM generated_subtitles WHERE videoid =? LIMIT 1'''
        self._cursor.execute(q_read_generated_subtitles, (video_id,))
        return self._cursor.fetchone() is not None

    def check_video_exists_in_subtitles(self, video_id: str) -> bool:
        q_read_subtitles = '''SELECT 1 FROM subtitles WHERE videoid =? LIMIT 1'''
        self._cursor.execute(q_read_subtitles, (video_id,))
        return self._cursor.fetchone() is not None

    def delete_subtitles_by_videoid(self, video_id: str):
        q_delete_subtitles = '''DELETE FROM subtitles WHERE videoid = ? '''
        self._cursor.execute(q_delete_subtitles, (video_id,))
        self._sqliteConnection.commit()

    def delete_generated_subtitles_by_videoid(self, video_id: str):
        q_delete_subtitles = '''DELETE FROM generated_subtitles WHERE videoid = ? '''
        self._cursor.execute(q_delete_subtitles, (video_id,))
        self._sqliteConnection.commit()

    def delete_sponsor_info_by_video_id(self,video_id: str):
        q_delete_sponsor = '''DELETE FROM sponsor_info WHERE videoid = ? '''
        self._cursor.execute(q_delete_sponsor, (video_id,))
        self._sqliteConnection.commit()

if __name__ == '__main__':
    my_sponsor_db = SponsorDB('data/SQLite_YTSP_subtitles.db')
    my_spi = SponsorInfo(video_id='test123', start_time=62.4, end_time=4003.7, upvotes=23,
                         downvotes=7)
    # my_sponsor_db.store_sponsor_info(my_spi)
    test = pd.DataFrame(my_sponsor_db.get_all_sponsor_info(),
                        columns=['uid', 'video_id', 'start_time', 'end_time', 'upvotes', 'downvotes'])
    test = test.sort_values('upvotes', ascending=False)
    test = test.reset_index(drop=True)
    print(test)
