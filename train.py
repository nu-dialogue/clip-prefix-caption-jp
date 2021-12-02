import torch
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import argparse
import sys
import os
import json
from tqdm import tqdm

from model import ClipDataset, build_model, bulid_model

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.model_name}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def train(train_dataset: Dataset, valid_dataset: Dataset, model, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    save_config(args)
    for epoch in range(epochs):
        print(f">>> Epoch: {epoch}")

        print(f"Training")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        model.train()
        for idx, (tokens, mask, prefix, _) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()

        print(f"Validation")
        sys.stdout.flush()
        progress = tqdm(total=len(valid_dataloader), desc=output_prefix)
        model.eval()
        for idx, (tokens, mask, prefix, _) in enumerate(valid_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, valid_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["coco", "sfcoco"])
    parser.add_argument('--model_name', help='prefix for saved filenames')
    parser.add_argument('--pretrained_path', default="")
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.set_defaults(only_prefix=False)
    args = parser.parse_args()

    prefix_length = args.prefix_length
    train_data_fpath = os.path.join("data", args.dataset, "exp", "train.pkl")
    valid_data_fpath = os.path.join("data", args.dataset, "exp", "valid.pkl")
    train_dataset = ClipDataset(train_data_fpath, prefix_length)
    valid_dataset = ClipDataset(valid_data_fpath, prefix_length)

    model = bulid_model(prefix_length=args.prefix_length,
                        only_prefix=args.only_prefix,
                        model_fpath=args.pretrained_path)
    
    train(train_dataset, valid_dataset, model, args, output_dir=args.out_dir, output_prefix=args.model_name)


if __name__ == '__main__':
    main()
