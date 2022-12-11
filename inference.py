import numpy as np
import torch
import skimage.io as io
from PIL import Image
from transformers import T5Tokenizer
from model import ClipCaptionModel, build_clip_model
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Predictor:
    def __init__(self, cap_model: ClipCaptionModel, cap_tokenizer: T5Tokenizer,
                 clip_model, clip_preprocess, device = DEVICE) -> None:
        self.model = cap_model
        self.model.eval()
        self.model.to(device)
        self.tokenizer = cap_tokenizer
        self.stop_token = cap_tokenizer.eos_token

        self.clip_model = clip_model
        self.preprocess = clip_preprocess

        self.device = device

    def caption(self, image_fpath, beam_size=5, prompt=None,):
        image = io.imread(image_fpath)
        pil_image = Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # if type(model) is ClipCaptionE2E:
            #     prefix_embed = model.forward_image(image)
            # else:
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(-1, self.model.prefix_length, self.model.gpt_embedding_size)
        generated_text_prefix = self.generate_beam(embed=prefix_embed, beam_size=beam_size)
        return pil_image, generated_text_prefix

    def generate_beam(self, embed, beam_size, prompt=None, entry_length=67, temperature=1.0):
        stop_token_index = self.tokenizer.encode(self.stop_token)[0]
        tokens = None
        scores = None
        device = next(self.model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(self.tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = self.model.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = self.model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.model.gpt.transformer.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [
            self.tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts