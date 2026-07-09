import numpy as np
import torch
import tiktoken
 
from gpt_download import download_and_load_gpt2
from model import GPTModel, GPT_CONFIG_124M, generate
from train import text_to_token_ids, token_ids_to_text

model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
# validate that our parameter and OpenAI's array have the same shape

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left (ours): {left.shape}, "
            f"Right (OpenAI): {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
 
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
 
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
 
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
 
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
 
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
 
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def show_next_token_probs(model, idx, tokenizer, top_k=5):
    with torch.no_grad():
        logits = model(idx)
 
    last_logits = logits[0, -1]
    probas = torch.softmax(last_logits, dim=-1)
 
    top_probas, top_ids = torch.topk(probas, top_k)
 
    print(f"\nTop {top_k} next-token predictions:")
    for prob, token_id in zip(top_probas, top_ids):
        token_str = tokenizer.decode([token_id.item()])
        print(f"  {token_str!r:15s} {prob.item()*100:.2f}%")

def main():
    
    settings, params = download_and_load_gpt2(
        model_size="774M", models_dir="gpt2"
    )
 
    
    model_name = "gpt2-large (774M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
 

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()
    load_weights_into_gpt(gpt, params)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    gpt.to(device)
 

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = input("Enter a prompt: ")
 
 #   show_next_token_probs(
 #       model=gpt,
 #       idx=text_to_token_ids(start_context, # tokenizer).to(device),
 #       tokenizer=tokenizer,
 #       top_k=5
 #   )

    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=50,
        context_size=NEW_CONFIG["context_length"],
        top_k=25,
        temperature=0.7
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
 
 
if __name__ == "__main__":
    main()