"""Group Relative Policy Optimisation with verifiable rewards."""

import argparse
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from medreason.common import load_config, load_model, load_tokenizer, lora_config, set_seed
from medreason.data import load_medqa
from medreason.rewards import build_reward_funcs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init", help="SFT adapter to start from; omit to run RL from the base model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    g = cfg["grpo"]
    tokenizer = load_tokenizer(cfg)
    train = load_medqa(cfg, tokenizer, "train")  # `answer` column is forwarded to the reward functions

    out = Path(cfg["output_root"]) / ("grpo" if args.init else "grpo_from_base")
    trainer = GRPOTrainer(
        model=load_model(cfg, adapters=[args.init] if args.init else None),
        reward_funcs=build_reward_funcs(cfg),
        args=GRPOConfig(
            output_dir=str(out),
            learning_rate=g["learning_rate"],
            num_generations=g["num_generations"],
            per_device_train_batch_size=g["batch_size"],
            gradient_accumulation_steps=g["gradient_accumulation_steps"],
            max_prompt_length=g["max_prompt_length"],
            max_completion_length=g["max_completion_length"],
            beta=g["beta"],
            temperature=g["temperature"],
            max_steps=g["max_steps"],
            use_vllm=g["use_vllm"],
            gradient_checkpointing=True,
            logging_steps=5,
            save_steps=100,
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
