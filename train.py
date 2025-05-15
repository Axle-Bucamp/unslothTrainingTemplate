import os
import torch
import wandb
from unsloth import FastLanguageModel, FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM
import argparse

# 🧾 Parse CLI arguments
parser = argparse.ArgumentParser(description="Generalized Unsloth trainer")
parser.add_argument("--model_path", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Local model path or HF model ID")
parser.add_argument("--dataset_path", type=str, default="dataset/enhance_train.parquet", help="Parquet dataset file")
parser.add_argument("--text_field", type=str, default="transformed_prompt", help="Dataset field to use for training")
parser.add_argument("--output_dir", type=str, default="outputs/final_model", help="Output model directory")
parser.add_argument("--max_steps", type=int, default=60, help="Max training steps")
parser.add_argument("--batch_size", type=int, default=10, help="Per device batch size")
parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
args = parser.parse_args()

# 📊 Initialize Weights & Biases
if args.use_wandb:
    wandb.init(
        project="unsloth-ollama-finetune",
        name=os.path.basename(args.output_dir)
    )

# 📦 Load dataset
dataset = load_dataset(
    "parquet",
    data_files={"train": args.dataset_path},
    split="train"
)

# 🧠 Load model with 4-bit quantization
model, tokenizer = FastModel.from_pretrained(
    model_name=args.model_path,
    max_seq_length=args.max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

# 🔍 Detect if model already has LoRA attached
has_lora = any("lora" in name.lower() for name, _ in model.named_parameters())

# 🔌 Apply LoRA if not already attached
if not has_lora:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )
else:
    print("✅ Model already has LoRA adapters. Skipping re-injection.")

# ⚙️ Set up SFT configuration
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        dataset_text_field=args.text_field,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=args.max_steps,
        logging_steps=1,
        output_dir=args.output_dir,
        optim="adamw_8bit",
        seed=3407,
        report_to="wandb" if args.use_wandb else None,
        save_strategy="no",
    ),
)

# 🚀 Train the model
trainer.train()

# 💾 Save model artifacts
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"✅ Training complete. Model saved to: {args.output_dir}")
