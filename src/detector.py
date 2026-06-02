from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RADAR_MODEL_NAME = "TrustSafeAI/RADAR-Vicuna-7B"
_radar_tokenizer = None
_radar_model = None
_device = None


def _load_radar():
    global _radar_tokenizer, _radar_model, _device
    if _radar_model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _radar_tokenizer = AutoTokenizer.from_pretrained(RADAR_MODEL_NAME, use_fast=True)
    _radar_model = AutoModelForSequenceClassification.from_pretrained(RADAR_MODEL_NAME)
    _radar_model.eval()
    _radar_model.to(_device)


@torch.no_grad()
def radar_score(text: str, max_length: int = 256):
    _load_radar()
    inputs = _radar_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(_device)

    out = _radar_model(**inputs)
    logits = out.logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)

    prob_ai = probs[0].item()
    prob_human = probs[1].item() if probs.numel() > 1 else (1.0 - prob_ai)
    return prob_ai, prob_human, logits.detach().cpu()
