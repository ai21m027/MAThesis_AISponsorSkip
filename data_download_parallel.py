import pandas as pd
import pickle
from subtitle_download import SubtitleToStorage
from rich.progress import track
pd.set_option('display.max_columns', None)
from multiprocessing.pool import Pool
from youtube_transcript_api import YouTubeTranscriptApi
import time

def youtube_download(video_id:str):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        english_transcript = transcript_list.find_manually_created_transcript(['en', 'en-US'])
    except:
        return None, None
    transcript = english_transcript.fetch()

    return transcript

if __name__=='__main__':
    """
    data = pd.read_feather('data/sponsorTimes.feather')
    data = data.sort_values('views',ascending=False)
    data = data.reset_index(drop=True)
    print(data.loc[0,'videoID'])
    print(data.loc[0:1000])
    """
    data = pd.read_feather('test_data/sponsorTimes_best1000.feather')
    #print(data.loc[0])
    storage_config={'type':'return_value'}
    #data = data.loc[:50]




    clean_data=pd.DataFrame(columns=['videoID','startTime','endTime','votes','incorrectVotes','subtitleID'])
    subtitle_ID = 0
    subtitle_dict = {}
    data_length = len(data)
    start = time.time()
    with Pool() as pool:
        ids = list(data.loc[:,'videoID'])
        #print(list(ids))
        results = pool.map(youtube_download,ids)

    #print(results)
    end = time.time()
    print(f"Parallel time: {end-start}")
    """
    start = time.time()
    result = []
    for i in range(len(data)):
        result.append(youtube_download(data.loc[i,'videoID']))
    end = time.time()
    print(f"Serial time: {end-start}")
    """

    """
    for i in track(range(data_length),description='Downloading subtitles'):
        start_time = data.loc[i,'startTime']
        end_time = data.loc[i,'endTime']
        downloader = SubtitleToStorage.SubtitlesToStorage(source='youtube',video_id=data.loc[i,'videoID'],storage_config=storage_config)
        subtitles,transcript = downloader.save()
        if subtitles is None:
            continue
        in_subtitle=False
        categorized_subtitles = []
        for element in transcript:
            if float(element['start']) >= start_time and float(element['start']) < end_time and not in_subtitle:
                in_subtitle=True
                categorized_subtitles.append({'text':element['text'],'category':'sponsored','start':element['start'],'duration':element['duration']})
                # if sponsor segment is only on transcript segment long set false immediatly
                if (float(element['start']) + float(element['duration'])) >= end_time and in_subtitle:
                    in_subtitle = False
            elif (float(element['start'])+float(element['duration'])) >= end_time and in_subtitle:
                in_subtitle = False
                #sponsor_text += element['text']
            elif in_subtitle:
                categorized_subtitles.append({'text':element['text'],'category':'sponsored','start':element['start'],'duration':element['duration']})
            else:
                categorized_subtitles.append({'text': element['text'], 'category': 'not-sponsored','start':element['start'],'duration':element['duration']})
        #print(categorized_subtitles)
        #print()
        df_catsub = pd.DataFrame(categorized_subtitles)
        subtitle_dict[subtitle_ID]=df_catsub
        short_data = data.loc[i,['videoID','startTime','endTime','votes','incorrectVotes']]
        short_data =short_data.to_frame().transpose()
        short_data['subtitleID'] = [subtitle_ID]
        clean_data = pd.concat([clean_data,short_data])
        clean_data = clean_data.reset_index(drop=True)
        clean_data.to_feather('data/current_progress.feather')
        with open('data/saved_subtitle_dictionary.pkl', 'wb') as f:
            pickle.dump(subtitle_dict, f)
        subtitle_ID+=1
    clean_data = clean_data.reset_index(drop=True)
    clean_data.to_feather('data/current_progress.feather')
    with open('data/saved_subtitle_dictionary.pkl', 'wb') as f:
        pickle.dump(subtitle_dict, f)
    print(clean_data.loc[3])
    print(subtitle_dict[clean_data.loc[3,'subtitleID']])

    """