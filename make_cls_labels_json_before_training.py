import pandas as pd 
import json

df = pd.read_csv("/home/jma/Documents/projects/meningiomas/train_with_nifti_remaining.csv")

mapping = dict(zip(df["renamed_nifti_file"], df["Aneurysm Present"]))

with open("/home/jma/Documents/projects/meningiomas/cls_labelsTr.json", "w") as f:
    json.dump(mapping, f, indent=4)