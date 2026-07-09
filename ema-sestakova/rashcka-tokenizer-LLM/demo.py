import torch
from tokenizer import RegexTokenizer
from model import GPTModel
from train import text_to_token_ids

# trained
model_path = "runs/run_2026-06-29_15-43-58/model_2026-06-29_15-43-58.pth"

# untrained
# model_path = "runs/run_2026-07-06_11-45-51/model_2026-07-06_11-45-51.pth"

max_generated = 100
top_k = 25
temperature = 0.8

def generate_stream(model, idx, max_new_tokens, context_size, 
                    temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float("-inf")).to(logits.device), 
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
        yield idx_next

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    tokenizer = RegexTokenizer()
    tokenizer.load(checkpoint["tokenizer_path"])
    return model, config, tokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    model, config, tokenizer = load_model(model_path, device)
    print("Model loaded.")
    print(f"Config: {config}\n")
    print("Type 'q' to exit.\n")
    while True:
        start_context = input("Input prompt: ")
        if start_context.lower() == 'q':
            break
        if start_context.strip() == "":
            continue
        input_ids = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            print("\nGenerated text:\n")
            print(start_context, end="", flush=True)
            for token_tensor in generate_stream(
                model=model,
                idx=input_ids,
                max_new_tokens=max_generated,
                context_size=config["context_length"],
                temperature=temperature,
                top_k=top_k
            ):
                token_id = token_tensor.squeeze(0).tolist()
                token_text = tokenizer.decode(token_id)
                print(token_text, end="", flush=True)
            print("\n" + "-" * 40)

if __name__ == "__main__":
    main()
