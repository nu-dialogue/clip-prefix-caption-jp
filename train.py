import torch
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import warnings
import argparse
import sys
import os
import json
from tqdm import tqdm

from model import ClipDataset, build_cap_model

def set_default_args_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--train_name_prefix', type=None, help='prefix for saved filenames')
    parser.add_argument('--dataset_name', type=str, help='preprocessed dataset')
    parser.add_argument('--rinna_gpt_name', type=str, default='gpt_medium', help='gpt_medium/gpt_1b')
    parser.add_argument('--clip_model_name', type=str, default='en_clip_b32', help='model name for clip')
    parser.add_argument('--pretrained_path', type=str, default=None)
    # parser.add_argument('--train_data_fpath', type=str)
    # parser.add_argument('--valid_data_fpath', type=str)
    parser.add_argument('--datasets_dpath', default='./data')
    parser.add_argument('--checkpoints_dpath', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=4)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=0)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--num_layers', type=int, default=8, help="number of transformer layers")
    parser.set_defaults(prefix_dim=512) # CLIP
    parser.add_argument('--n_gpu', type=int, default=1)

def make_train_name(args: argparse.Namespace):
    elems = []
    if args.train_name_prefix:
        elems.append(args.train_name_prefix)
    elems += [args.dataset_name,
              args.rinna_gpt_name,
              args.clip_model_name,
              args.mapping_type,
              "prefix" if args.only_prefix else "finetune",
              f"ep{args.epochs}",
              f"bs{args.train_batch_size}",
              f"lr{args.lr}"]
    return "-".join(elems)

def save_config(args: argparse.Namespace, output_dir: str):
    out_path = os.path.join(output_dir, "args.json")
    with open(out_path, 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    if device_count < args.n_gpu:
        warnings.wart(f"n_gpu is set to {device_count} because "
                      f"the specified number of GPUs {args.n_gpu} is not available")
        args.n_gpu = device_count

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    print(f"Number of GPU: {args.n_gpu}")
    print(f"Total train batch size: {args.train_batch_size}")

    train_name = make_train_name(args)
    print(f"Train name: {train_name}")

    dataset_dir = os.path.join(args.datasets_dpath, args.dataset_name, f"processed-{args.clip_model_name}")
    args.train_data_fpath = os.path.join(dataset_dir, "train.pkl")
    args.valid_data_fpath = os.path.join(dataset_dir, "valid.pkl")

    output_dir = os.path.join(args.checkpoints_dpath, train_name)
    os.makedirs(output_dir, exist_ok=True)
    save_config(args, output_dir=output_dir)

    train_dataset = ClipDataset(args.train_data_fpath, args.prefix_length)
    valid_dataset = ClipDataset(args.valid_data_fpath, args.prefix_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=True, drop_last=True)

    
    model, tokenizer = build_cap_model(rinna_gpt_name=args.rinna_gpt_name,
                                       clip_model_name=args.clip_model_name,
                                       prefix_length=args.prefix_length,
                                       prefix_length_clip=args.prefix_length_clip,
                                       prefix_dim=args.prefix_dim,
                                       num_layers=args.num_layers,
                                       mapping_type=args.mapping_type,
                                       only_prefix=args.only_prefix,
                                       pretrained_path=args.pretrained_path)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    log = []
    
    for epoch in range(args.epochs):
        print(f">>> Epoch: {epoch}")

        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=f"Training...")
        model.train()
        losses = []
        epoch_log = {"epoch": epoch}
        for idx, (tokens, mask, prefix, _) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

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
                    os.path.join(output_dir, f"latest.pt"),
                )
        progress.close()
        print(f"Training avg loss: {sum(losses)/len(losses)}")
        epoch_log["train_avg_loss"] = sum(losses)/len(losses)

        sys.stdout.flush()
        progress = tqdm(total=len(valid_dataloader), desc=f"Evaluating...")
        model.eval()
        losses = []
        for idx, (tokens, mask, prefix, _) in enumerate(valid_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, valid_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            if args.n_gpu > 1:
                loss = loss.mean()
            progress.set_postfix({"loss": loss.item()})
            losses.append(loss.item())
            progress.update()
        progress.close()
        print(f"Validation avg loss: {sum(losses)/len(losses)}")
        epoch_log["valid_avg_loss"] = sum(losses)/len(losses)

        if (args.save_every > 0) and ((epoch+1) % args.save_every == 0 or (epoch+1) == args.epochs):
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{epoch:03d}.pt"),
            )
        log.append(epoch_log)
        json.dump(log, open(os.path.join(output_dir, "log.json"), "w"), indent=4)
    if log:
        best_epoch = sorted(log, key = lambda x: x["valid_avg_loss"])[0]["epoch"]
        best_pt_fpath = os.path.join(output_dir, f"{best_epoch:03d}.pt")
    else:
        best_pt_fpath = None
    return output_dir, best_pt_fpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_default_args_to_parser(parser=parser)
    train(args = parser.parse_args())
