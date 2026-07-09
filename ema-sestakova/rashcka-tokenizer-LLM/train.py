"""
Pretraining.
"""
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tokenizer import RegexTokenizer
from model import GPTModel, generate
from dataset import prepare_dataloaders, print_dataset, load_text


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, # original dropout: 0.1
    "qkv_bias": False
}

GPT_CONFIG_SMALL = {
    "vocab_size": 5000,
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.2,
    "qkv_bias": False
}
# ----------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    # .unsqueeze(0) adds the batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    # removes batch dimension
    return tokenizer.decode(flat.tolist())

# calculates the cross entropy loss of a given batch returned via the
# training and validation loader
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# computes the training and validation loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # iteratives over all batches if no fixed num_batches is specified
        num_batches = len(data_loader)
    else:
        # reduces the number of batches to match the total number of batches in the data
        # loader if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # sums loss for each batch
            total_loss += loss.item()
        else:
            break
    # averages loss over all batches
    return total_loss / num_batches

# prints the training and validation set losses after each model update
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # dropot is disabled during evaluation for stable, reproducible results
    model.eval()
    # disables gradient tracking
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

# the main function for pretraining LLMs
def train_model(model, train_loader, val_loader, optimizer, device, 
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer,
                       accumulation_steps=1):
    # initializes lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # main training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss = loss / accumulation_steps
            # calculate loss gradients
            loss.backward()
            tokens_seen += input_batch.numel()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                # OPTIONAL evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                    )
        # handle leftover gradients if the number of batches isn't divisible
        # by accumulation_steps
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

# generates a text sample
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# plots training and validation set losses
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    # create new folder for runs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    # find and load text file
    file_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    raw_text = load_text(file_path)
    # find and load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "shakespeare-tokenizer/shakespeare_tokenizer.model")   
    tokenizer = RegexTokenizer()
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("No existing tokenizer found, training a new one")
        tokenizer.train(raw_text, vocab_size=GPT_CONFIG_SMALL["vocab_size"], verbose=True)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer_prefix = tokenizer_path[:-len(".model")] 
        tokenizer.save(tokenizer_prefix)
        print(f"Saved new tokenizer to {tokenizer_path}")
    train_loader, val_loader = prepare_dataloaders(
        file_path=file_path,
        tokenizer=tokenizer,
        batch_size=24, # original batch_size=2
        max_length=GPT_CONFIG_SMALL["context_length"],
        stride=128, # original stride=256
        train_ratio=0.9,
        num_workers=0
    )
    stats = print_dataset(file_path, tokenizer, train_ratio=0.9)
    print("----- Dataset stats -----")
    print(f"Training tokens: {stats['train_tokens']}")
    print(f"Validation tokens: {stats['val_tokens']}")
    print(f"Number of unique characters: {len(stats['unique_chars'])}")
    print("Unique characters:")
    print(stats["unique_chars"])
    print("-------------------------")
    model = GPTModel(GPT_CONFIG_SMALL).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.2
        # original lr=0.0004
        # original weight_decay=0.1
    )
    # paths
    model_path = os.path.join(run_dir, f"model_{timestamp}.pth")
    plot_path = os.path.join(run_dir, f"loss_plot_{timestamp}.png")
    num_epochs = 20
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs,
        eval_freq=10, # changed from 5
        eval_iter=20, # changed from 5
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        accumulation_steps=4
        )
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, save_path=plot_path)
    # save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": GPT_CONFIG_SMALL,
        "tokenizer_path": tokenizer_path,
    }, model_path)
    print(f"Final model saved as {model_path}")
    model.eval()
    context = text_to_token_ids("Every effort moves you", tokenizer).to(device)
    token_ids = generate(
        model=model,
        idx=context,
        max_new_tokens=15,
        context_size=GPT_CONFIG_SMALL["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    main()
