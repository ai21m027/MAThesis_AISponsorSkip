import pandas as pd
import pickle



with open('data/saved_subtitle_dictionary.pkl', 'rb') as f:
    subtitles_dict = pickle.load(f)

data = pd.read_feather('data/current_progress.feather')
print(subtitles_dict)
print(data)

print(len(data))
print(len(subtitles_dict))