import pandas as pd
import argparse
import random
import torch
import time
import requests
from tqdm import tqdm
# from unsloth import FastLanguageModel

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
    },
    {
        "instruction": "Explain the cause-and-effect relationship within the text.",
        "header": "Cause and effect analysis:"
    },
    {
        "instruction": "Create a logical argument based on the given text.",
        "header": "Logical argument:"
    },
    {
        "instruction": "Form an analogy related to the text '{{text}}'.",
        "header": "Analogy:"
    },
    {
        "instruction": "Identify any biases present in the text '{{text}}'.",
        "header": "Bias detection:"
    },
    {
        "instruction": "Evaluate the strengths and weaknesses of the argument presented in the text.",
        "header": "Argument evaluation:"
    },
    {
        "instruction": "Generate a counter-argument to the claims in the text '{{text}}'.",
        "header": "Counter-argument:"
    },
    {
        "instruction": "Classify the text into one of the logical fallacies.",
        "header": "Fallacy detection:"
    },
    {
        "instruction": "List the pros and cons based on the text.",
        "header": "Pros and cons:"
    },
    {
        "instruction": "Summarize the text '{{text}}' with an emphasis on reasoning.",
        "header": "Reasoning summary:"
    },
    {
        "instruction": "Offer a solution or recommendation based on the text's content.",
        "header": "Solution/recommendation:"
    },
    {
        "instruction": "Critically analyze the validity of the information presented in the text '{{text}}'.",
        "header": "Validity analysis:"
    },
    {
        "instruction": "Propose an alternative perspective to the ideas in the text.",
        "header": "Alternative perspective:"
    },
    {
        "instruction": "Analyze the text's logical structure and coherence.",
        "header": "Logical structure analysis:"
    },
    {
        "instruction": "Detect any contradictions within the text '{{text}}'.",
        "header": "Contradiction detection:"
    },
    {
        "instruction": "Break down the reasoning in the text into steps.",
        "header": "Reasoning breakdown:"
    },
    {
        "instruction": "Generate a thought experiment related to the content of the text.",
        "header": "Thought experiment:"
    },
    {
        "instruction": "Identify patterns or trends in the arguments presented in the text.",
        "header": "Pattern recognition:"
    },
    {
        "instruction": "Discuss the implications of the text’s claims.",
        "header": "Implication discussion:"
    },
    {
        "instruction": "Assess the credibility of the sources mentioned in the text.",
        "header": "Source credibility assessment:"
    }
]

# -------------------------------
# 🚀 Load Unsloth Model
# -------------------------------
'''
def load_unsloth_model(model_name):
    max_seq_length = 8000
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    return model, tokenizer
'''

# -------------------------------
# 🌟 Transform Text Functions
# -------------------------------
def transform_text_unsloth(model, tokenizer, text, instruction, header):
    max_total_tokens = 2048
    max_new_tokens = 200
    token_budget = max_total_tokens - max_new_tokens

    instruction = instruction.replace("{{text}}", text[:300])  # Template tag replacement

    system_prompt = f"<|system|>\n{instruction}<|end|>\n"
    assistant_prompt = "<|assistant|>"
    user_prefix = "<|user|>\n"
    user_suffix = "<|end|>\n"

    budget_left = token_budget - sum(len(tokenizer(t, return_tensors="pt")["input_ids"][0]) for t in [system_prompt, assistant_prompt, user_prefix, user_suffix])
    user_input_tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    if len(user_input_tokens) > budget_left:
        user_input_tokens = user_input_tokens[:budget_left]
        text = tokenizer.decode(user_input_tokens, skip_special_tokens=True)

    full_prompt = f"{system_prompt}{user_prefix}{text}{user_suffix}{assistant_prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=token_budget).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"{header}\n{result.split('<|assistant|>')[-1].strip()}"

def transform_text_ollama(text, instruction, header, model_name="llama3"):
    instruction = instruction.replace("{{text}}", text[:300])
    full_prompt = f"{instruction}\n\n{text.strip()}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": full_prompt, "stream": False}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    result = response.json()["response"].strip()
    return f"{header}\n{result}"

# -------------------------------
# 📦 Process Parquet
# -------------------------------
def process_parquet(input_file, column, output_file, model_name, engine="ollama", sample_output=None, sample_count=5):
    df = pd.read_parquet(input_file)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    tqdm.pandas(desc="✨ Transforming Text")
    
    model = tokenizer = None
    if engine == "unsloth":
        pass
        # model, tokenizer = load_unsloth_model(model_name)

    def apply_transformation(text):
        option = random.choice(TRANSFORMATION_OPTIONS)
        def call():
            if engine == "unsloth":
                return transform_text_unsloth(model, tokenizer, text, option["instruction"], option["header"])
            return transform_text_ollama(text, option["instruction"], option["header"], model_name)
        result = safe_transform(call)
        return {
            "prompt": result,
            "instruction": option["instruction"],
            "header": option["header"]
        }

    results = df[column].progress_apply(lambda x: apply_transformation(str(x)))
    df["transformed_prompt"] = results.apply(lambda x: x["prompt"])
    df["used_instruction"] = results.apply(lambda x: x["instruction"])
    df["used_header"] = results.apply(lambda x: x["header"])

    df.to_parquet(output_file, index=False)
    print(f"\n📄 Saved transformed dataset to: {output_file}")

    if sample_output:
        df.head(sample_count)[["transformed_prompt", "used_instruction"]].to_csv(sample_output, index=False)
        print(f"📝 Sample written to {sample_output}")

# -------------------------------
# ⚠️ Retry-safe Wrapper
# -------------------------------
def safe_transform(transform_fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return transform_fn()
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)
    return "[ERROR] Failed to transform after retries"

# -------------------------------
# 💾 CLI Entry
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform Parquet text with reasoning prompts.")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--column", required=True, help="Column to transform")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--model", default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Model name or path")
    parser.add_argument("--engine", choices=["ollama", "unsloth"], default="ollama", help="Engine to use")
    parser.add_argument("--sample-output", type=str, help="Optional CSV file for preview")
    parser.add_argument("--sample-count", type=int, default=5, help="How many samples to write")
    args = parser.parse_args()

    start = time.time()
    process_parquet(args.input, args.column, args.output, args.model, args.engine, args.sample_output, args.sample_count)
    print(f"\n🕒 Done in {time.time() - start:.2f}s")
