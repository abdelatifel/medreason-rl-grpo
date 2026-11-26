"""Direct Preference Optimisation on pairs sampled from the SFT policy."""

import argparse
from pathlib import Path

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

from medreason.common import load_config, load_model, load_tokenizer, lora_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init", required=True, help="SFT adapter to start from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    d = cfg["dpo"]
    tokenizer = load_tokenizer(cfg)
    pairs = load_dataset("json", data_files=str(Path(cfg["output_root"]) / "preferences.jsonl"), split="train")

    out = Path(cfg["output_root"]) / "dpo"
    trainer = DPOTrainer(
        model=load_model(cfg, adapters=[args.init]),
        ref_model=None,  # with a peft_config, the reference is the merged SFT model with the new adapter disabled
        args=DPOConfig(
            output_dir=str(out),
            num_train_epochs=d["epochs"],
            learning_rate=d["learning_rate"],
            beta=d["beta"],
            per_device_train_batch_size=d["batch_size"],
            gradient_accumulation_steps=d["gradient_accumulation_steps"],
            max_length=d["max_length"],
            max_prompt_length=d["max_prompt_length"],
            gradient_checkpointing=True,
            logging_steps=10,
            seed=cfg["seed"],
            report_to="none",
        ),
        train_dataset=pairs,
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(out))


if __name__ == "__main__":
    main()
