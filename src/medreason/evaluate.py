"""Accuracy, format compliance and answer-position bias on the MedQA test set."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from medreason.common import load_config, load_model, load_tokenizer, set_seed
from medreason.data import LETTERS, load_medqa
from medreason.rewards import FORMAT_RE, extract_answer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def summarise(golds: list[str], outputs: list[str]) -> dict:
    preds = [extract_answer(o) for o in outputs]
    n = len(golds)
    return {
        "accuracy": sum(p == g for p, g in zip(preds, golds)) / n,
        "valid_format_rate": sum(bool(FORMAT_RE.match(o)) for o in outputs) / n,
        "mean_completion_words": sum(len(o.split()) for o in outputs) / n,
        "predicted_letter_dist": {k: Counter(preds)[k] / n for k in LETTERS},
        "gold_letter_dist": {k: Counter(golds)[k] / n for k in LETTERS},
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--adapter", nargs="+", default=[],
        help="adapters merged in order, e.g. outputs/sft outputs/grpo; omit for the base model",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, adapters=args.adapter).eval()

    test = load_medqa(cfg, tokenizer, "test")
    if args.limit:
        test = test.select(range(args.limit))

    outputs = [generate(model, tokenizer, ex["prompt"], cfg["eval"]["max_new_tokens"]) for ex in tqdm(test)]
    metrics = summarise(test["answer"], outputs)
    print(json.dumps(metrics, indent=2))

    name = "+".join(Path(a).name for a in args.adapter) or "base"
    out = Path(cfg["output_root"]) / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "completions.jsonl").open("w", encoding="utf-8") as f:
        for gold, o in zip(test["answer"], outputs):
            f.write(json.dumps({"gold": gold, "output": o}) + "\n")


if __name__ == "__main__":
    main()
