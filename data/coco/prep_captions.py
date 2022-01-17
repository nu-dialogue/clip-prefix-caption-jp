import json
import csv

with open("yjcaptions26k_clean.json") as f:
    data = json.load(f)["annotations"]

new_data = []
for d in data:
    img_id = d["image_id"]
    image_name = f"COCO_train2014_{int(img_id):012d}.jpg"
    new_data.append([image_name, d["caption"]])

with open("captions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(new_data)