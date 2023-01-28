import sqlite3
from typing import NamedTuple
from datetime import datetime

my_db = 'data/SQLite_YTSP_subtitles.db'


class SponsorInfo(NamedTuple):
    video_id: str
    start_time: datetime
    end_time: datetime
    upvotes: int
    downvotes: int


def init_database(db_name: str):
    try:
        # if database does not exist sqlite3.connect creates the database
        sqliteConnection = sqlite3.connect(db_name)
        cursor = sqliteConnection.cursor()
        print(f"Successfully Connected to DB {db_name}")

        # check for table video_info
        q_check_table = '''SELECT name FROM sqlite_master
                        WHERE type = 'table'
                        AND name = 'sponsor_info';
                        '''
        cursor.execute(q_check_table)
        if not len(cursor.fetchall()):  # if table does not exist, create table
            q_create_table = '''CREATE TABLE sponsor_info (
                                uid INTEGER PRIMARY KEY,
                                videoid TEXT NOT NULL,
                                startTime datetime NOT NULL,
                                endTime datetime NOT NULL,
                                upvotes INTEGER NOT NULL,
                                downvotes INTEGER NOT NULL);'''
            cursor.execute(q_create_table)
            sqliteConnection.commit()
            print('Table sponsor_info created')
        else:
            print('Table sponsor_info exists')
        cursor.close()

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("The SQLite connection is closed")


def store_sponsor_info(s_storage:str,s_info:SponsorInfo):
    sqliteConnection = sqlite3.connect(storage_name)


if __name__ == '__main__':
    init_database(my_db)

"""
sqlite_create_table_query = '''CREATE TABLE SqliteDb_developers2 (
                            id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            email text NOT NULL UNIQUE,
                            joining_date datetime,
                            salary REAL NOT NULL);'''
cursor.execute(sqlite_create_table_query)
sqliteConnection.commit()
"""
