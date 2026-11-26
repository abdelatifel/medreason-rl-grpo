"""Verifiable reward functions for GRPO.

Each function follows the TRL convention: it receives the batch of completions plus
any dataset columns as keyword arguments, and returns one float per completion.
"""

import re

ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>\s*[A-D]\s*</answer>\s*$", re.DOTALL | re.IGNORECASE)


def extract_answer(completion: str) -> str | None:
    match = ANSWER_RE.search(completion)
    return match.group(1).upper() if match else None


def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    return [1.0 if extract_answer(c) == gold else 0.0 for c, gold in zip(completions, answer)]


def make_format_reward(weight: float):
    def format_reward(completions: list[str], **kwargs) -> list[float]:
        return [weight if FORMAT_RE.match(c) else 0.0 for c in completions]

    return format_reward


def make_length_penalty(budget_words: int, penalty_per_word: float):
    def length_penalty(completions: list[str], **kwargs) -> list[float]:
        return [-penalty_per_word * max(0, len(c.split()) - budget_words) for c in completions]

    return length_penalty


def build_reward_funcs(cfg: dict) -> list:
    r = cfg["rewards"]
    return [
        correctness_reward,
        make_format_reward(r["format_weight"]),
        make_length_penalty(r["length_budget"], r["length_penalty_per_word"]),
    ]
