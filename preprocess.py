import torch
import skimage.io as io
import math
from PIL import Image
import pickle
import json
import os
import csv
import argparse
from tqdm import tqdm
import random
from model import build_clip_model

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse(data, part, clip_model, preprocess, image_dpath, out_dpath):
    if not data:
        print(f"No {part} data is prepared.")
        return

    all_names = []
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = {
            "clip_embedding": i,
            "image_id": data[i][0],
            "image_name": data[i][1],
            "caption": data[i][2]
        }

        fpath = os.path.join(image_dpath, d["image_name"])
        if not os.path.isfile(fpath):
            raise FileNotFoundError(fpath)

        image = io.imread(fpath)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        all_captions.append(d)
        all_names.append(d["image_name"])

    out_data = {"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}
    data_fpath = os.path.join(out_dpath, f"{part}.pkl")
    pickle.dump(out_data, open(data_fpath, "wb"))
    json.dump(all_names, open(os.path.join(out_dpath, f"{part}_list.json"), "w"), indent=4)
    print(f"Saved {part} data to {data_fpath}.")
    return data_fpath

def prepare_data(clip_model_name, captions_fpath, image_dpath, test_ratio, valid_ratio, train_ratio, shuffle=False):
    """
    [
        {
            "caption": ***,
            "id": ***,
            "image_name": ***
        },
        ...
    ]
    """
    assert sum([test_ratio,valid_ratio,train_ratio]) <= 1.
    clip_model, preprocess = build_clip_model(clip_model_name)

    out_dpath = os.path.join(os.path.dirname(captions_fpath), f"processed-{clip_model_name}")
    if not os.path.exists(out_dpath):
        os.makedirs(out_dpath)

    all_data = [[i] + line for i, line in enumerate(csv.reader(open(captions_fpath)))]
    if shuffle:
        all_data = random.sample(all_data, len(all_data))

    test_size = math.ceil(len(all_data) * test_ratio)
    valid_size = math.ceil(len(all_data) * valid_ratio)
    train_size = math.ceil(len(all_data) * train_ratio)
    print(f"{len(all_data)} captions loaded from json.")
    print(f"\ttrain size: {train_size}\n\tvalid size: {valid_size}\n\ttest size: {test_size}")

    test_data_fpath = parse(data=all_data[:test_size], part="test",
                            image_dpath=image_dpath, out_dpath=out_dpath,
                            clip_model=clip_model, preprocess=preprocess)
    valid_data_fpath = parse(data=all_data[test_size:test_size+valid_size], part="valid",
                             image_dpath=image_dpath, out_dpath=out_dpath,
                             clip_model=clip_model, preprocess=preprocess)
    train_data_fpath = parse(data=all_data[test_size+valid_size:test_size+valid_size+train_size], part="train",
                             image_dpath=image_dpath, out_dpath=out_dpath,
                             clip_model=clip_model, preprocess=preprocess)
                             
    return test_data_fpath, valid_data_fpath, train_data_fpath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model_name", type=str, help="en_clip_b32, ja_clip_b16, ja_cloob_b16")
    parser.add_argument("--captions_fpath", type=str)
    parser.add_argument("--image_dpath", type=str)
    args = parser.parse_args()
    coco_test_fpath, coco_valid_fpath, coco_train_fpath = prepare_data(clip_model_name=args.clip_model_name,
                                                                       captions_fpath=args.captions_fpath,
                                                                       image_dpath=args.image_dpath,
                                                                       test_ratio=0.1,
                                                                       valid_ratio=0.1,
                                                                       train_ratio=0.8,
                                                                       shuffle=False)