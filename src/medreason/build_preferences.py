"""Build DPO preference pairs by sampling from the SFT policy.

For each question we sample k completions and pair a correct, well-formatted one
(chosen) with an incorrect one (rejected). Questions where all samples agree carry no
preference signal and are skipped.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from medreason.common import load_config, load_model, load_tokenizer, set_seed
from medreason.data import load_medqa
from medreason.rewards import FORMAT_RE, extract_answer


@torch.no_grad()
def sample(model, tokenizer, prompt: str, k: int, temperature: float, max_new_tokens: int) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens,
        num_return_sequences=k, pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", required=True, help="adapter to sample from (the SFT output)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    p = cfg["preferences"]
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, adapters=[args.policy]).eval()

    train = load_medqa(cfg, tokenizer, "train").shuffle(seed=cfg["seed"])
    train = train.select(range(min(p["n_questions"], len(train))))

    pairs = []
    for ex in tqdm(train, desc="sampling"):
        completions = sample(model, tokenizer, ex["prompt"], p["samples_per_question"], p["temperature"], p["max_new_tokens"])
        good = [c for c in completions if extract_answer(c) == ex["answer"] and FORMAT_RE.match(c)]
        bad = [c for c in completions if extract_answer(c) != ex["answer"]]
        if good and bad:
            pairs.append({"prompt": ex["prompt"], "chosen": random.choice(good), "rejected": random.choice(bad)})

    out = Path(cfg["output_root"]) / "preferences.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"{len(pairs)} preference pairs from {len(train)} questions -> {out}")


if __name__ == "__main__":
    main()
