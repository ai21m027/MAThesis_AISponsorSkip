import pandas as pd
from subtitle_download import SubtitleToStorage
pd.set_option('display.max_columns', None)


if __name__=='__main__':
    """
    data = pd.read_feather('data/sponsorTimes.feather')
    data = data.sort_values('views',ascending=False)
    data = data.reset_index(drop=True)
    print(data.loc[0,'videoID'])
    print(data.loc[0:1000])
    """
    data = pd.read_feather('data/sponsorTimes_best1000.feather')
    print(data.loc[0])
    storage_config={'type':'return_value'}

    clean_data=pd.DataFrame(columns=['videoID','startTime','endTime','votes','incorrectVotes','subtitleID'])
    subtitle_ID = 0
    subtitle_dict = {}
    for i in range(20):
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
                categorized_subtitles.append({'text':element['text'],'category':'sponsored'})
                # if sponsor segment is only on transcript segment long set false immediatly
                if (float(element['start']) + float(element['duration'])) >= end_time and in_subtitle:
                    in_subtitle = False
            elif (float(element['start'])+float(element['duration'])) >= end_time and in_subtitle:
                in_subtitle = False
                #sponsor_text += element['text']
            elif in_subtitle:
                categorized_subtitles.append({'text':element['text'],'category':'sponsored'})
            else:
                categorized_subtitles.append({'text': element['text'], 'category': 'not-sponsored'})
        #print(categorized_subtitles)
        print()
        df_catsub = pd.DataFrame(categorized_subtitles)
        subtitle_dict[subtitle_ID]=df_catsub
        short_data = data.loc[i,['videoID','startTime','endTime','votes','incorrectVotes']]
        short_data =short_data.to_frame().transpose()
        short_data['subtitleID'] = [subtitle_ID]
        clean_data = pd.concat([clean_data,short_data])
        #print(clean_data)
        #print(subtitle_dict)
        subtitle_ID+=1
    print(clean_data)
    print(subtitle_dict)
    #print(transcript)