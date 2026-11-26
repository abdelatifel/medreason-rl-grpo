"""Supervised warm-start: teach the output format and the answer distribution."""

import argparse
from pathlib import Path

from trl import SFTConfig, SFTTrainer

from medreason.common import load_config, load_model, load_tokenizer, lora_config, set_seed
from medreason.data import load_medqa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)

    # MedQA has no reference rationales, so SFT targets carry a minimal <think> block;
    # the reasoning itself is left for the RL stage to shape.
    train = load_medqa(cfg, tokenizer, "train").map(
        lambda ex: {"completion": f"<think>Considering each option.</think>\n<answer>{ex['answer']}</answer>"},
        remove_columns=["answer"],
    )

    s = cfg["sft"]
    out = Path(cfg["output_root"]) / "sft"
    trainer = SFTTrainer(
        model=load_model(cfg),
        args=SFTConfig(
            output_dir=str(out),
            num_train_epochs=s["epochs"],
            learning_rate=s["learning_rate"],
            per_device_train_batch_size=s["batch_size"],
            gradient_accumulation_steps=s["gradient_accumulation_steps"],
            max_length=s["max_length"],
            completion_only_loss=True,
            gradient_checkpointing=True,
            logging_steps=10,
            seed=cfg["seed"],
            report_to="none",
        ),
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(out))


if __name__ == "__main__":
    main()
