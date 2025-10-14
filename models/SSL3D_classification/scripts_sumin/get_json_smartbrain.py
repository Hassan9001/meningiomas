import os
import glob
import numpy as np
import json
from pathlib import Path
import pandas as pd

path = '/home/sumin/Documents/SmartBrain/CNN-LSTM/scripts/tables/AD_diagnosis_train_test_NoPrediction.csv'
df = pd.read_csv(os.path.join(path))
df = df[['file', 'Finl_Diag_AD', 'split']]
#df['file'] =df['file'].apply(lambda x: '__'.join(x.split('__')[:2]))
df['file'] = df['file'].apply(lambda x: '__'.join(x.split('__')[:2]) + '/ses-DEFAULT/' + x.split('.')[0])
dest_path = '/home/sumin/Documents/nnssl_test/nnssl_data/nnssl_preprocessed/Dataset002_SmartBRAINMRI'

# df_img = {
#     im: int(df[df['EVLP_ID'].apply(lambda x: x in im)]['disposition_left'].values[0])
#     for im in img_files
# }
df_tr = df[df['split']=='train']
df_test = df[df['split']=='test']
df_img = {
    im: int(df_tr[df_tr['file']==im]['Finl_Diag_AD'].values[0])
    for im in df_tr['file'].tolist() 
}
df_img_train = {
    im: int(df_test[df_test['file']==im]['Finl_Diag_AD'].values[0])
    for im in df_test['file'].tolist() 
}

df_img_train = {
    im: int(df[df['file']==im]['Finl_Diag_AD'].values[0])
    for im in df[df['split']=='train']['file'].tolist() 
}
tr_imgs = df_tr['file'].tolist()
tst_imgs = df_test['file'].tolist()
splits_final = {
    'train': tr_imgs,
    'val': tst_imgs
}
with open(os.path.join(dest_path, 'splits_final.json'), 'w') as f:
    #f.write(df_json)
    json.dump(splits_final, f, indent=4)

df_json = df_img
print(df_json)
with open(os.path.join(dest_path, 'labelsTr.json'), 'w') as f:
    #f.write(df_json)
    json.dump(df_json, f, indent=4)