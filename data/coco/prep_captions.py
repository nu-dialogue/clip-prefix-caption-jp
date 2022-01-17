import json

with open("yjcaptions26k_clean.json") as f:
    data = json.load(f)["annotations"]

for d in data:
    img_id = d["image_id"]
    d["image_name"] = f"COCO_train2014_{int(img_id):012d}.jpg"

with open("captions.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)