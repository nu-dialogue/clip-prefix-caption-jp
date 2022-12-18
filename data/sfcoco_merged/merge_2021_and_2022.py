import json
import pickle
import os
import csv
import shutil

def load_captions(csv_fpath):
    with open(csv_fpath) as f:
        rows = list(csv.reader(f))
    return rows

captions_2021 = load_captions("../sfcoco2021/captions.csv")
captions_2022 = load_captions("../sfcoco2022/captions.csv")

os.makedirs("images", exist_ok=True)
captions_merged = []
for image_fname, caption_text in captions_2021:
    captions_merged.append([f"2021-{image_fname}", caption_text])
    shutil.copy2(f"../sfcoco2021/images/{image_fname}", f"images/2021-{image_fname}")

for image_fname, caption_text in captions_2022:
    captions_merged.append([f"2022-{image_fname}", caption_text])
    shutil.copy2(f"../sfcoco2022/images/{image_fname}", f"images/2022-{image_fname}")

with open("captions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(captions_merged)