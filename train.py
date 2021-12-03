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

from model import ClipDataset, build_model

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.checkpoints_dpath, f"{args.model_name}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def train(train_dataset: Dataset, valid_dataset: Dataset, model,
          batch_size, epochs, lr, warmup_steps: int = 5000,
          save_every=1, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs):
        print(f">>> Epoch: {epoch}")

        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=f"{output_prefix} train")
        model.train()
        losses = []
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
            losses.append(loss.item())
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        print(f"{output_prefix} train avg loss: {sum(losses)/len(losses)}")

        sys.stdout.flush()
        progress = tqdm(total=len(valid_dataloader), desc=f"{output_prefix} valid")
        model.eval()
        losses = []
        for idx, (tokens, mask, prefix, _) in enumerate(valid_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, valid_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            progress.set_postfix({"loss": loss.item()})
            losses.append(loss.item())
            progress.update()
        progress.close()
        print(f"{output_prefix} valid avg loss: {sum(losses)/len(losses)}")

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='prefix for saved filenames')
    parser.add_argument('--pretrained_fpath', default="")
    parser.add_argument('--train_data_fpath', default='data/coco/outputs/train.pkl')
    parser.add_argument('--valid_data_fpath', default='data/coco/outputs/valid.pkl')
    parser.add_argument('--checkpoints_dpath', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.set_defaults(only_prefix=False)
    args = parser.parse_args()

    train_dataset = ClipDataset(args.train_data_fpath, args.prefix_length)
    valid_dataset = ClipDataset(args.valid_data_fpath, args.prefix_length)

    model = build_model(prefix_length=args.prefix_length,
                        only_prefix=args.only_prefix,
                        model_fpath=args.pretrained_fpath)

    if not os.path.exists(args.checkpoints_dpath):
        os.makedirs(args.checkpoints_dpath)
    save_config(args)

    train(train_dataset=train_dataset, valid_dataset=valid_dataset, model=model,
          batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
          save_every=args.save_every, output_dir=args.checkpoints_dpath, output_prefix=args.model_name)