import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Path to your fine-tuned model directory
MODEL_PATH = "outputs/final_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # or bfloat16 if preferred
)

# Create generation pipeline (🚫 DO NOT set `device` manually)
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.1,
)

# Interactive Q&A
print("🔍 Model ready. Ask anything (type 'exit' to quit):\n")

while True:
    query = input("🧠 You: ")
    if query.lower().strip() in {"exit", "quit"}:
        print("👋 Exiting...")
        break

    prompt = f"<|user|>\n{query}\n<|assistant|>\n"
    response = qa_pipeline(prompt)[0]["generated_text"]

    # Parse response to exclude prompt
    reply = response.split("<|assistant|>")[-1].strip() if "<|assistant|>" in response else response.strip()
    print(f"🤖 AI: {reply}\n")
