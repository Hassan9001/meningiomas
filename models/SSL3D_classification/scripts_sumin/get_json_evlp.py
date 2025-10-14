import os
import glob
import numpy as np
import json
from pathlib import Path
import pandas as pd

path = '/home/sumin/Documents/cryoSumin/nnUNet_data/nnUNet_raw/Dataset416_Jun'
dest_path = '/home/sumin/Documents/cryoSumin/nnUNet_data/nnUNet_preprocessed/Dataset416_Jun'

img_files = os.listdir('/home/sumin/Documents/cryoSumin/nnUNet_data/nnUNet_raw/Dataset416_Jun/imagesTr')
img_files = [x for x in img_files if x.endswith('.png')]
img_files = [i.split('_0000.')[0] for i in img_files]



df = pd.read_csv(os.path.join(path, 'labels.csv'))
df = df[['EVLP_ID','disposition_left']]
#df_json = df.to_json(orient='records')
df['EVLP_ID'] = df['EVLP_ID'].apply(lambda x: "EVLP" + str(x).zfill(4) )
#df_dict = dict(zip(df["EVLP_ID"], df["disposition_left"]))
# df_img = {
#     im: df[df['EVLP_ID'].str.contains(im)]['disposition_left'].values[0]
#     for im in img_files
# }
df_img = {
    im: int(df[df['EVLP_ID'].apply(lambda x: x in im)]['disposition_left'].values[0])
    for im in img_files
}

print(df_img)
df_json = df_img

print(df_json)
with open(os.path.join(dest_path, 'labelsTr.json'), 'w') as f:
    #f.write(df_json)
    json.dump(df_json, f)