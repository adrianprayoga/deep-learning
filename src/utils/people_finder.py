"""
Simple function to go through the corrupted json file and extract the needed information
"""

# %%
import pandas as pd
import os
import re
import json

# %%
fp = '/home/ginger/code/gderiddershanghai/deep-learning/data/DFDB_Face_metadata.json' 
# print(os.path.exists(fp))
with open(fp, 'r') as f:
    raw_data = f.read()


# %%
cleaned_data = re.sub(r',,+', ',', raw_data)
# %%
fp_cleam = '/home/ginger/code/gderiddershanghai/deep-learning/data/DFDB_Face_metadata_cleaned.json' 

with open(fp_cleam, 'w') as f:
    f.write(cleaned_data)
# %%
df = pd.read_json(fp_cleam, lines=True)
# %%
csv_fp = '/home/ginger/code/gderiddershanghai/deep-learning/data/cleaned_DFDB_Face_metadata.csv'
segments = cleaned_data.split('image_name')

names = []
prompts = []
nsfws = []

cleaned_segments = []
for idx, segment in enumerate(segments[1:]): 
    all = segment.split(':')
    # print(all)
    name = all[1].split(',')[0]
    prompt = all[2].split(',')[0]
    nsfw = all[-2].split(',')[0]
    names.append(name)
    prompts.append(prompt)
    nsfws.append(nsfw)
    # print(cleaned_segment)
    # break
df = pd.DataFrame({'name': names, 'prompt': prompts, 'nsfw': nsfws})
# %%
df.to_csv(csv_fp, index=False)
# %%
