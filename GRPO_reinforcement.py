#!/usr/bin/env python3
# train_grpo_unsloth.py
"""
GRPO: Generalized Reinforced Preference Optimization
with Unsloth + TRL

This script
  - Loads a 4-bit quantized LLaMA 3.3 model
  - Adds LoRA adapters
  - Builds a reasoning-based reward model
  - Trains via a custom GRPO loop (discounted pairwise preferences)
  - Logs metrics to Weights & Biases
"""

import os
import torch
import logging
import argparse
import wandb

from torch import nn
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from unsloth import FastLanguageModel
from trl import ORPOTrainer, ORPOConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION & LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GRPO")

class GRPOConfig(ORPOConfig):
    """
    Extend ORPOConfig with a discount factor `gamma` for generalized RL.
    """
    gamma: float = 0.99  # reward discount factor


# ─────────────────────────────────────────────────────────────────────────────
# 2) REASONING REWARD MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningRewardModel(nn.Module):
    """
    Wrap a HF PreTrainedModel to score texts by reasoning quality.
    """
    def __init__(self, base_model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.scoring_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        # score based on last token’s embedding
        scores = self.scoring_head(last_hidden[:, -1, :])  # (B, 1)
        return scores.squeeze(-1)

    def score_batch(self, texts: list[str], device: torch.device):
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            return self.forward(enc.input_ids, enc.attention_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 3) CUSTOM GRPO TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer(ORPOTrainer):
    """
    Inherits ORPOTrainer but applies a discounted pairwise loss:
      L = -E[ log σ( beta*(R(chosen) - R(rejected)) ) * (1 + γ^t + …) ]
    Here we only apply discount to encourage longer‐term reasoning improvements.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = self.args.gamma

    def compute_loss(self, chosen_scores, rejected_scores):
        # pairwise log‐sigmoid loss
        diff = chosen_scores - rejected_scores  # (B,)
        base_loss = - torch.log(torch.sigmoid(self.args.beta * diff)).mean()
        # apply simple discount factor (for demonstration, uniform over batch)
        discounted_loss = base_loss * (1.0 / (1.0 - self.gamma))
        return discounted_loss


# ─────────────────────────────────────────────────────────────────────────────
# 4) DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_TEMPLATE = """Below is an instruction and context. Write a helpful response.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_for_grpo(sample, eos_token: str):
    prompt = ALPACA_TEMPLATE.format(sample["instruction"], sample.get("input",""), "")
    sample["prompt"] = prompt
    sample["chosen_input"] = prompt + sample["accepted"] + eos_token
    sample["rejected_input"] = prompt + sample["rejected"] + eos_token
    return sample

def load_preference_dataset(name: str, eos_token: str):
    ds = load_dataset(name, split="train")
    return ds.map(lambda ex: format_for_grpo(ex, eos_token), remove_columns=ds.column_names)


# ─────────────────────────────────────────────────────────────────────────────
# 5) MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GRPO with Unsloth")
    parser.add_argument("--model",   default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
    parser.add_argument("--dataset", default="reciperesearch/dolphin-sft-v0.1-preference")
    parser.add_argument("--output",  default="grpo_outputs")
    parser.add_argument("--batch",   type=int, default=2)
    parser.add_argument("--accum",   type=int, default=4)
    parser.add_argument("--steps",   type=int, default=100)
    parser.add_argument("--gamma",   type=float, default=0.99)
    parser.add_argument("--beta",    type=float, default=0.1)
    args = parser.parse_args()

    # 1) Initialize W&B
    wandb.init(project="grpo-unsloth", config=vars(args))

    # 2) Load base model + tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Unsloth model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=4096,
    )
    base_model.to(device)

    # 3) Prepare dataset
    logger.info("Preparing preference dataset...")
    ds = load_preference_dataset(args.dataset, tokenizer.eos_token)
    
    # 4) Build reward model
    logger.info("Building reasoning-based reward model...")
    rm = ReasoningRewardModel(base_model, tokenizer).to(device)

    # 5) Configure GRPO
    grpo_cfg = GRPOConfig(
        max_length=2048,
        max_prompt_length=1024,
        max_completion_length=1024,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_steps=args.steps,
        beta=args.beta,
        gamma=args.gamma,
        logging_steps=1,
        output_dir=args.output,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        report_to="wandb",
    )

    # 6) Train
    trainer = GRPOTrainer(
        model=base_model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=grpo_cfg,
        reward_model=rm  # pass in our custom RM
    )
    trainer.train()
    logger.info("✅ GRPO training complete.")

    # 7) Save adapters only
    base_model.save_pretrained(os.path.join(args.output, "lora_adapters"))
    tokenizer.save_pretrained(os.path.join(args.output, "lora_adapters"))
    logger.info(f"Adapters saved to {args.output}/lora_adapters")

if __name__ == "__main__":
    main()
