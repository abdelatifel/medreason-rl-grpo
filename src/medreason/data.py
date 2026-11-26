"""MedQA loading and the shared prompt template."""

from datasets import Dataset, load_dataset

LETTERS = ("A", "B", "C", "D")

INSTRUCTION = (
    "You are a medical expert answering a USMLE-style question. "
    "Think step by step inside <think></think>, then give only the letter of the "
    "correct option inside <answer></answer>."
)


def format_question(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{letter}. {options[letter]}" for letter in LETTERS)
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nOptions:\n{opts}\n"


def to_prompt(tokenizer, question: str, options: dict[str, str]) -> str:
    messages = [{"role": "user", "content": format_question(question, options)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_medqa(cfg: dict, tokenizer, split: str) -> Dataset:
    """Return a dataset with `prompt` and `answer` (gold letter) columns."""
    ds = load_dataset(cfg["data"]["dataset"], split=split)
    limit = cfg["data"].get("max_train_examples")
    if split == "train" and limit:
        ds = ds.shuffle(seed=cfg["seed"]).select(range(limit))
    return ds.map(
        lambda ex: {"prompt": to_prompt(tokenizer, ex["question"], ex["options"]), "answer": ex["answer_idx"]},
        remove_columns=ds.column_names,
    )
