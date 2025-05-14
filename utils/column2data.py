import pandas as pd
import argparse
from tqdm import tqdm
import random
import torch
from unsloth import FastLanguageModel

# -------------------------------
# 🔁 Transformation Options
# -------------------------------
TRANSFORMATION_OPTIONS = [
    {
        "instruction": "Summarize this text clearly.",
        "header": "Summarize the following:"
    },
    {
        "instruction": "Turn the text into a question starting with 'What is... ?'",
        "header": "Answer the question:"
    },
    {
        "instruction": "Rewrite the text in fill-in-the-blank format.",
        "header": "Fill in the blanks:"
    },
    {
        "instruction": "Paraphrase this text with simpler words.",
        "header": "Rephrase this text:"
    },
    {
        "instruction": "Explain this text as if teaching a beginner.",
        "header": "Explain this text:"
    },
    {
        "instruction": "Convert this text into a multiple-choice quiz format.",
        "header": "Create a quiz from the text:"
    },
    {
        "instruction": "Enhance and rewrite this document with more detail.",
        "header": "Enhance this document:"
    }
]

# -------------------------------
# 🚀 Load Unsloth Model
# -------------------------------
def load_unsloth_model(model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"):
    """
    Load the Unsloth model and tokenizer with optimal settings.
    """
    max_seq_length = 8000
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    return model, tokenizer

# -------------------------------
# ✨ Apply Transformation
# -------------------------------
def transform_text_unsloth(model, tokenizer, text: str, instruction: str, header: str) -> str:
    """
    Apply the transformation instruction using Unsloth and return formatted prompt.
    """
    max_total_tokens = 8192
    max_new_tokens = 200
    token_budget = max_total_tokens - max_new_tokens

    system_prompt = f"<|system|>\n{instruction}<|end|>\n"
    assistant_prompt = "<|assistant|>"
    user_prefix = "<|user|>\n"
    user_suffix = "<|end|>\n"

    # Tokenize and calculate budget
    system_tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"][0]
    assistant_tokens = tokenizer(assistant_prompt, return_tensors="pt")["input_ids"][0]
    prefix_tokens = tokenizer(user_prefix, return_tensors="pt")["input_ids"][0]
    suffix_tokens = tokenizer(user_suffix, return_tensors="pt")["input_ids"][0]

    budget_left = token_budget - len(system_tokens) - len(assistant_tokens) - len(prefix_tokens) - len(suffix_tokens)

    user_input_tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if len(user_input_tokens) > budget_left:
        user_input_tokens = user_input_tokens[:budget_left]
        text = tokenizer.decode(user_input_tokens, skip_special_tokens=True)

    # Build full prompt
    full_prompt = f"{system_prompt}{user_prefix}{text}{user_suffix}{assistant_prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=token_budget).to(model.device)

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip the assistant marker
    if "<|assistant|>" in result:
        result = result.split("<|assistant|>")[-1].strip()

    return f"{header}\n{result.strip()}"

# -------------------------------
# 📦 Process Parquet File
# -------------------------------
def process_parquet(input_file: str, column: str, output_file: str, model_name: str):
    """
    Apply a random transformation to each entry in the specified column of a Parquet file.
    """
    df = pd.read_parquet(input_file)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in Parquet file.")

    print("📥 Loading model...")
    model, tokenizer = load_unsloth_model(model_name)

    tqdm.pandas(desc="✨ Transforming Text")

    def apply_random_transformation(text):
        option = random.choice(TRANSFORMATION_OPTIONS)
        return transform_text_unsloth(model, tokenizer, str(text), instruction=option["instruction"], header=option["header"])

    print("⚙️ Applying transformations...")
    df["transformed_prompt"] = df[column].progress_apply(apply_random_transformation)

    df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved transformed dataset to: {output_file}")

# -------------------------------
# 🧾 CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply random prompt transformations using Unsloth.")
    parser.add_argument("--input", required=True, help="Input Parquet file path.")
    parser.add_argument("--column", required=True, help="Column name to transform.")
    parser.add_argument("--output", required=True, help="Output Parquet file path.")
    parser.add_argument("--model", default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Model name or local path.")

    args = parser.parse_args()
    process_parquet(args.input, args.column, args.output, args.model)
