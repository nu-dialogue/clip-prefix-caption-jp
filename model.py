import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    T5Tokenizer
)

from glob import glob
import sys
import os
import pickle
from typing import Tuple, Optional, Union
from enum import Enum
import json
from collections import OrderedDict

import clip as en_clip
import japanese_clip as ja_clip

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ClipDataset(Dataset):
    def __init__(self, data_path: str,  prefix_length: int, rinna_gpt_name: str = "rinna/japanese-gpt2-medium"):
        self.tokenizer = T5Tokenizer.from_pretrained(rinna_gpt_name)
        self.tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        self.prefix_length = prefix_length
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        # self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        return tokens, mask, self.prefixes[self.caption2embedding[item]], self.captions[item]

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class ClipCaptionModel(nn.Module):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, rinna_gpt_path: str = "rinna/japanese-gpt2-medium"):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained(rinna_gpt_path)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[mapping_type]
        if mapping_type == MappingType.MLP:
            mlp_sizes = (prefix_size,
                         (self.gpt_embedding_size * prefix_length) // 2,
                         self.gpt_embedding_size * prefix_length)
            self.clip_project = MLP(sizes=mlp_sizes)
        elif mapping_type == MappingType.Transformer:
            self.clip_project = TransformerMapper(dim_clip=prefix_size, dim_embedding=self.gpt_embedding_size, prefix_length=prefix_length,
                                                  clip_length=clip_length, num_layers=num_layers)
        else:
            raise ValueError(mapping_type)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):

    # def parameters(self, for_optimizer: bool = False, recurse: bool = True):
    #     return self.clip_project.parameters()

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, rinna_gpt_path: str = "rinna/japanese-gpt2-medium"):
        super().__init__(prefix_length, clip_length, prefix_size, num_layers, mapping_type, rinna_gpt_path)
        for param in self.gpt.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def process_pretrained_path(pretrained_path):
    if os.path.isdir(pretrained_path):
        pretrained_dpath = pretrained_path
        last_epoch = "000"
        latest_exists = False
        for checkpoint_fpath in glob(os.path.join(pretrained_path, "*.pt")):
            epoch = os.path.splitext(os.path.basename(checkpoint_fpath))[0]
            if int(last_epoch) < int(epoch):
                last_epoch = epoch
            if epoch == "latest":
                latest_exists = True
        checkpoint_fpath = os.path.join(pretrained_path, f"{epoch}.pt")
    elif os.path.isfile(pretrained_path):
        pretrained_dpath = os.path.dirname(pretrained_path)
        checkpoint_fpath = pretrained_path
    else:
        raise FileNotFoundError(pretrained_path)
    args_fpath = os.path.join(pretrained_dpath, "args.json")
    return pretrained_dpath, checkpoint_fpath, args_fpath
    

def build_cap_model(rinna_gpt_name, clip_model_name, prefix_length, prefix_length_clip, prefix_dim, num_layers, mapping_type, only_prefix, pretrained_path=None):
    if rinna_gpt_name == "gpt_medium":
        rinna_gpt_path = "rinna/japanese-gpt2-medium"
    elif rinna_gpt_name == "gpt_1b":
        rinna_gpt_path = "rinna/japanese-gpt-1b"
    else:
        raise NotImplementedError(rinna_gpt_name)

    if only_prefix:
        model = ClipCaptionPrefix(prefix_length=prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=num_layers, mapping_type=mapping_type, rinna_gpt_path=rinna_gpt_path)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length=prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=num_layers, mapping_type=mapping_type, rinna_gpt_path=rinna_gpt_path)
        print("Train both prefix and GPT")

    if pretrained_path:
        _, checkpoint_fpath, args_fpath = process_pretrained_path(pretrained_path)
        # argument check
        pretrained_args = json.load(open(args_fpath))
        assert pretrained_args["rinna_gpt_name"] == rinna_gpt_name
        assert pretrained_args["clip_model_name"] == clip_model_name
        assert pretrained_args["prefix_length"] == prefix_length
        assert pretrained_args["prefix_length_clip"] == prefix_length_clip
        assert pretrained_args["prefix_dim"] == prefix_dim
        assert pretrained_args["num_layers"] == num_layers
        assert pretrained_args["mapping_type"] == mapping_type

        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
        state_dict = torch.load(checkpoint_fpath, map_location=torch.device("cpu"))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):] # remove `module.`
            new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)
        print(f"Resume pretrained weights from {checkpoint_fpath}")

    tokenizer = T5Tokenizer.from_pretrained(rinna_gpt_path)
    return model, tokenizer

def build_clip_model(clip_model_name, device = DEVICE):
    if clip_model_name == "en_clip_b32":
        clip_model, preprocess = en_clip.load("ViT-B/32", device=device, jit=False)
    elif clip_model_name == "ja_clip_b16":
        clip_model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", device=device)
    elif clip_model_name == "ja_cloob_b16":
        clip_model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
    else:
        raise NotImplementedError(clip_model_name)
    return clip_model, preprocess

def build_models_from_pretrained(pretrained_path, device = DEVICE):
    _, _, args_fpath = process_pretrained_path(pretrained_path)
        
    args = json.load(open(args_fpath))
    cap_model, cap_tokenizer = build_cap_model(rinna_gpt_name=args["rinna_gpt_name"],
                                               clip_model_name=args["clip_model_name"],
                                               prefix_length=args["prefix_length"],
                                               prefix_length_clip=args["prefix_length_clip"],
                                               prefix_dim=args["prefix_dim"],
                                               num_layers=args["num_layers"],
                                               mapping_type=args["mapping_type"],
                                               only_prefix=args["only_prefix"],
                                               pretrained_path=pretrained_path)
    clip_model, clip_preprocess = build_clip_model(clip_model_name=args["clip_model_name"],
                                                   device=device)
    return cap_model, cap_tokenizer, clip_model, clip_preprocess