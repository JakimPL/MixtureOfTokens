import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_vanilla import VanillaTransformer

from transformer_mot import MixtureOfTokens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4

N_SAMPLES = 16
N_EPOCHS = 5
LEARNING_RATE = 1e-4

D_MODEL = 256
N_HEAD = 4
N_LAYERS = 6
D_FF = 512


def get_dataloader(input_ids: torch.Tensor) -> DataLoader:
    dataset = torch.utils.data.TensorDataset(input_ids)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int = N_EPOCHS,
        device: torch.device = DEVICE
):
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in tqdm(range(n_epochs), position=0, leave=True):
        total_loss = 0
        total_perplexity = 1
        start_time = time.time()
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)

            targets = inputs[:, 1:].contiguous().view(-1)
            outputs = outputs[:, :-1, :].contiguous().view(-1, model.vocab_size)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_perplexity = torch.exp(loss).item()

        end_time = time.time()
        average_loss = total_loss / len(dataloader)
        average_perplexity = total_perplexity / len(dataloader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}, Perplexity: {average_perplexity}")

        history.append({
            "epoch": epoch,
            "loss": average_loss,
            "perplexity": average_perplexity,
            "time": end_time - start_time
        })

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The model: either 'vanilla' or 'mot'.")
    parser.add_argument("input", type=str, help="The input file path where the tokenized tensor is saved.")
    parser.add_argument("--path", type=str, help="The model path where the trained model will be saved.")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="The number of epochs")
    args = parser.parse_args()

    input_path = args.input
    model_path = args.path
    n_epochs = args.epochs

    dataset = torch.load(input_path)
    input_ids = dataset["input_ids"].to(DEVICE)
    vocab_size = dataset["vocab_size"]
    dataloader = get_dataloader(input_ids)

    if args.model == "vanilla":
        model = VanillaTransformer(vocab_size=vocab_size, d_model=D_MODEL, nhead=N_HEAD, d_ff=D_FF, num_layers=N_LAYERS).to(DEVICE)
    elif args.model == "mot":
        model = MixtureOfTokens(vocab_size=vocab_size, d_model=D_MODEL, nhead=N_HEAD, d_ff=D_FF, num_layers=N_LAYERS).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, dataloader, optimizer, n_epochs=n_epochs)

    if model_path:
        torch.save(model.state_dict(), model_path)
