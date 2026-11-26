"""Model loading shared by every training stage."""

import random

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: dict, adapters: list[str] | None = None):
    """Load the base model and merge any earlier stages' adapters into it, in order.

    Merging lets the next stage train a fresh LoRA on top of the previous one, and lets
    evaluation rebuild exactly the weights a stage was trained on (e.g. [sft, grpo]).
    """
    adapters = adapters or []
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant = None
    if cfg["model"].get("load_in_4bit", True) and not adapters:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    # When stacking adapters we load in half precision, because merging into 4-bit weights is lossy.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], quantization_config=quant, torch_dtype=dtype, device_map="auto"
    )
    for path in adapters:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return model


def lora_config(cfg: dict) -> LoraConfig:
    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    )
