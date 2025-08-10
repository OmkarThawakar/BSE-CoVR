import datetime
import time
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump


class TestWebVidCoVR:
    def __init__(self, remove_self_similarity=True):
        self.remove_self_similarity = remove_self_similarity
        
    def encode_prompt(self, model, prompt):
        prompt_tokens = model.tokenizer(
            prompt,
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(model.text_encoder.device)
        
        encoder_input_ids = prompt_tokens.input_ids.clone()
        encoder_input_ids[:, 0] = model.tokenizer.enc_token_id

        prompt_output = model.text_encoder(
            encoder_input_ids,
            attention_mask=prompt_tokens.attention_mask,
            return_dict=True,
            mode=None
        )
        prompt_embeds = prompt_output.last_hidden_state[:, 0, :]
        return model.prompt_proj(prompt_embeds)

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        
        query_feats = []

        for ref_img, img_prompt, caption, _, *_ in data_loader:

            device = ref_img.device

            ref_img_embs = model.visual_encoder(ref_img)
            
            prompt_embs = self.encode_prompt(model, img_prompt)
            combined_embs = ref_img_embs + prompt_embs.unsqueeze(1)
            combined_atts = torch.ones(combined_embs.size()[:-1], dtype=torch.long).to(device)

            text = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
            query_embs = model.text_encoder(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=combined_embs,
                encoder_attention_mask=combined_atts,
                return_dict=True,
            )
            query_feat = query_embs.last_hidden_state[:, 0, :]
            query_feat = F.normalize(model.text_proj(query_feat), dim=-1)
            query_feats.append(query_feat.cpu())

        return query_feats