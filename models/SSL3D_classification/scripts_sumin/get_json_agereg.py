import os
import glob
import numpy as np
import json
from pathlib import Path
import pandas as pd

path = '/home/sumin/Documents/nnssl_test/nnssl_data/src/openneuro_ds004856/train_label.csv'
df = pd.read_csv(os.path.join(path))
df = df[['subject_id' ,'AGE']]
#df['file'] =df['file'].apply(lambda x: '__'.join(x.split('__')[:2]))
#df['file'] = df['file'].apply(lambda x: '__'.join(x.split('__')[:2]) + '/ses-DEFAULT/' + x.split('.')[0])
dest_path = '/home/sumin/Documents/nnssl_test/nnssl_data/nnssl_preprocessed/Dataset003_AgeReg'

#img_files = 
# df_img = {
#     im: int(df[df['EVLP_ID'].apply(lambda x: x in im)]['disposition_left'].values[0])
#     for im in df['subject_id'].tolist()
# }
img_files = {
    im: int(df[df['subject_id']==im]['AGE'].values[0])
    for im in df['subject_id'].tolist()
}

#train and test split
train_ratio = 0.8
img_list = list(img_files.keys())
np.random.shuffle(img_list)
split_index = int(len(img_list) * train_ratio)
train_imgs = img_list[:split_index]
test_imgs = img_list[split_index:]
df_tr = {im: img_files[im] for im in train_imgs}
df_test = {im: img_files[im] for im in test_imgs}
splits_final = {
    'train': list(df_tr.keys()),
    'val': list(df_test.keys())
}
with open(os.path.join(dest_path, 'splits.json'), 'w') as f:
#     #f.write(df_json)
    json.dump(splits_final, f, indent=4)
# df_tr = df[df['split']=='train']
# df_test = df[df['split']=='test']
# df_img = {
#     im: int(df_tr[df_tr['file']==im]['Finl_Diag_AD'].values[0])
#     for im in df_tr['file'].tolist() 
# }
# df_img_train = {
#     im: int(df_test[df_test['file']==im]['Finl_Diag_AD'].values[0])
#     for im in df_test['file'].tolist() 
# }

# df_img_train = {
#     im: int(df[df['file']==im]['Finl_Diag_AD'].values[0])
#     for im in df[df['split']=='train']['file'].tolist() 
# }
# tr_imgs = df_tr['file'].tolist()
# tst_imgs = df_test['file'].tolist()
# splits_final = {
#     'train': tr_imgs,
#     'val': tst_imgs
# }
# with open(os.path.join(dest_path, 'splits_final.json'), 'w') as f:
#     #f.write(df_json)
#     json.dump(splits_final, f, indent=4)

df_json = img_files
print(df_json)
with open(os.path.join(dest_path, 'labelsTr.json'), 'w') as f:
    #f.write(df_json)
    json.dump(df_json, f, indent=4)