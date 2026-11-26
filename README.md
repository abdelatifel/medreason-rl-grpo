# MedReason-RL: Reinforcement Learning with Verifiable Rewards for Medical Reasoning in Small LLMs

Does **reinforcement learning** improve medical reasoning beyond supervised fine-tuning? This project trains a small open LLM on USMLE-style multiple-choice questions and compares three ways of post-training it:

| Method | Signal | Library |
|---|---|---|
| **SFT** | imitate the reference answer | `trl.SFTTrainer` |
| **DPO** | prefer correct sampled answers over incorrect ones | `trl.DPOTrainer` |
| **GRPO** | on-policy RL with **verifiable rewards** (correctness + format) | `trl.GRPOTrainer` |

GRPO (Group Relative Policy Optimization) is the algorithm behind DeepSeek-R1. It needs no learned reward model: it samples a group of answers per question and reinforces the ones that score above the group mean. Multiple-choice medicine is a good fit because correctness can be **checked automatically**.

---

## Reward design

The model has to reason inside `<think>...</think>` and then commit to `<answer>X</answer>`.

| Reward | Value | Purpose |
|---|---|---|
| `correctness_reward` | 1.0 if the answer letter matches the key, else 0 | the actual objective |
| `format_reward` | 0.2 if the output follows the `<think>/<answer>` template | makes answers parseable |
| `length_penalty` | small penalty beyond a token budget | limits rambling and reward hacking through verbosity |

## Pipeline

```
MedQA (USMLE, 4 options)
      │
      ├── SFT ─────────────────────────────► model_sft
      │                                        │
      │        sample k answers per question   │
      │        correct vs. incorrect pairs ◄───┤
      │                    │                   │
      │                   DPO ───────────────► model_sft_dpo
      │                                        │
      └──────────── GRPO (verifiable rewards) ► model_sft_grpo
```

The SFT stage can reuse the adapter from [`medllm-qlora-finetune`](https://github.com/abdelatifel/medllm-qlora-finetune).

## Results

> Filled in after running `scripts/run_all.sh`. Accuracy on the MedQA 4-option test set (1,273 questions).

| Model | Accuracy | Valid-format rate | Mean completion length |
|---|---|---|---|
| Base (zero-shot) | – | – | – |
| SFT | – | – | – |
| SFT + DPO | – | – | – |
| SFT + GRPO | – | – | – |

Training curves (reward, KL divergence, completion length) are logged to `outputs/<run>/`.

## Failure-mode analysis

RL on small models is known to fail in specific ways. The evaluation script checks for:

- **Reward hacking:** valid format with empty or copy-pasted reasoning
- **Answer-position bias:** distribution of predicted letters vs. the true distribution
- **Reasoning–answer inconsistency:** the `<think>` block argues for one option and `<answer>` picks another (spot-checked manually on a sample)

## Quickstart

```bash
pip install -e .

python -m medreason.sft   --config configs/qwen1.5b.yaml
python -m medreason.build_preferences --config configs/qwen1.5b.yaml --policy outputs/sft
python -m medreason.dpo   --config configs/qwen1.5b.yaml --init outputs/sft
python -m medreason.grpo  --config configs/qwen1.5b.yaml --init outputs/sft
python -m medreason.evaluate --config configs/qwen1.5b.yaml --adapter outputs/sft outputs/grpo
```

GRPO is the expensive step: with `Qwen2.5-1.5B-Instruct`, LoRA and 8 generations per prompt, expect several hours on a single 24 GB GPU. Set `grpo.max_steps` in the config for a shorter run.

## Repository structure

```
configs/qwen1.5b.yaml      all hyperparameters
src/medreason/
  data.py                  MedQA loading and prompt template
  rewards.py               verifiable reward functions
  sft.py                   supervised warm-start
  build_preferences.py     sample from the SFT policy, build (chosen, rejected) pairs
  dpo.py                   direct preference optimisation
  grpo.py                  group relative policy optimisation
  evaluate.py              accuracy, format rate, position bias
tests/test_rewards.py      reward functions are unit-tested; they *are* the objective
```

## Data and ethics

- **MedQA** (USMLE-style questions) is public. **No patient data.**
- Accuracy on exam questions **does not** mean clinical competence. This is a study of training methods, not a clinical tool.

## License

MIT
