import sys  # System-specific parameters and functions
import datasets  # Library for accessing datasets
from typing import List, Tuple, Mapping  # Type hinting for better code readability
import numpy as np  # NumPy for numerical operations
from tqdm import tqdm  # Progress bar library for iterations
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimizer module from PyTorch for model training
import torch.nn.functional as F  # Functional operations like activation functions and loss calculations
from torch.nn.utils.rnn import pad_sequence  # Utility for padding sequences in RNNs
from torch.utils.data import Dataset, DataLoader  # Dataset and DataLoader for efficient data handling and batching

# Set a manual seed for reproducibility
torch.manual_seed(42)

# Load the dataset
dataset = datasets.load_dataset("benjamin/ner-uk")

# Define token and padding IDs
TOK_PAD_ID = 0  # Define a token ID for padding input sequences.
NER_PAD_ID = -100  # Padding ID for NER tags

# Initialize vocabulary with padding token
vocab = {"<PAD>": TOK_PAD_ID}
curr_idx = 1

# Iterate through dataset splits (train, validation, test) to create vocabulary
for split in ("train", "validation", "test"):
    for sample in dataset[split]:
        for word in sample["tokens"]:
            if word not in vocab:
                vocab[word] = curr_idx  # Add the word to the vocabulary with the current index.
                curr_idx += 1  # Increment the current index.

print("Vocab size:", len(vocab))


class NERDataset(Dataset):
    def __init__(self, samples: datasets.Dataset, vocabulary: Mapping[str, int]) -> None:
        """
        Initialize a Named Entity Recognition (NER) dataset.

        Args:
            samples (datasets.Dataset): The dataset containing samples with tokens and NER tags.
            vocabulary (Mapping[str, int]): A mapping from tokens to their corresponding indices.
        """
        self.samples = samples  # Store the input samples dataset.
        self.vocabulary = vocabulary  # Store the vocabulary for token-to-index mapping.

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Retrieve a sample and its corresponding NER tags by index.

        Args:
            index (int): The index of the desired sample.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: A tuple containing token indices and NER tags.
        """
        sample = self.samples[index]  # Get the sample at the specified index.

        # Convert tokens to their corresponding indices using the vocabulary.
        doc = torch.LongTensor([self.vocabulary[token] for token in sample["tokens"]])
        label = torch.LongTensor(sample["ner_tags"])  # Convert NER tags to LongTensor.

        return doc, label  # Return the token indices and NER tags for the sample.


def seq_collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]], data_pad: int, label_pad: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Combine samples into a batch for the RNN model.

    Args:
        batch: List of tensors to pack into a batch.
            Each sample should be a tuple of (text_tokens, label_tokens).
        data_pad: Value used for padding text tokens.
        label_pad: Value used for padding label tokens.

    Returns:
        Padded tensors for text tokens and label tokens.
    """
    token_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=data_pad)
    label_ids = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=label_pad)
    return token_ids, label_ids


def ner_collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Collator function for the NER dataset.

    Args:
        batch: List of tensors to pack into a batch.
            Each sample should be a tuple of (text_tokens, label_tokens).

    Returns:
        Padded tensors for text tokens and label tokens.
    """
    return seq_collate_fn(batch, TOK_PAD_ID, NER_PAD_ID)


def sequence_f1(true_labels: np.array, predicted_labels: np.array) -> np.array:
    """Calculate F1 score for one sequence.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Model predictions.

    Returns:
        F1 scores for each class.
    """
    assert len(true_labels) == len(predicted_labels), "Mismatched length between true labels and predicted labels"

    scores = []
    for _cls in targets:
        true_positives = np.sum((true_labels == predicted_labels) & (true_labels == _cls))  # Correct positive predictions
        false_positives = np.sum((true_labels != predicted_labels) & (predicted_labels == _cls))  # Incorrect positive predictions
        false_negatives = np.sum((true_labels != predicted_labels) & (true_labels == _cls))  # Missed positive predictions

        precision = np.nan_to_num(true_positives / (true_positives + false_positives), nan=0.0)  # Calculate precision
        recall = np.nan_to_num(true_positives / (true_positives + false_negatives), nan=0.0)  # Calculate recall
        f1_score = np.nan_to_num(2 * (precision * recall) / (precision + recall), nan=0.0)  # Calculate F1 score

        scores.append(f1_score)
    return np.array(scores)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    verbose: bool = True,
) -> Mapping[str, np.array]:
    """Train model for one epoch.

    Args:
        model: Model to train.
        loader: DataLoader to use for training.
        criterion: Loss function to optimize.
        optimizer: Model training algorithm.
        device: Device to use for training (default is "cpu").
        verbose: Option to print training progress (default is True).

    Returns:
        Dictionary containing training logs.
    """
    model.train()

    losses = []
    scores = []

    with tqdm(total=len(loader), desc="training", file=sys.stdout, ncols=100, disable=not verbose) as progress:
        for x_batch, y_true in loader:
            x_batch = x_batch.to(device)  # Long tensor [B, T]
            y_true = y_true.to(device)    # Long tensor [B, T]

            optimizer.zero_grad()

            log_prob = model(x_batch)

            B, T = y_true.shape
            loss = criterion(log_prob.view(B * T, -1), y_true.view(B * T))

            loss.backward()
            losses.append(loss.item())

            y_pred = log_prob.argmax(2).detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            padding_mask = y_true != NER_PAD_ID
            for i in range(x_batch.size(0)):
                scores.append(sequence_f1(y_true[i][padding_mask[i]], y_pred[i][padding_mask[i]]))

            progress.set_postfix_str(f"loss {losses[-1]:.4f}")

            optimizer.step()
            progress.update(1)

    logs = {
        "losses": np.array(losses),
        "f1": np.array(scores)
    }
    return logs


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    verbose: bool = True,
) -> Mapping[str, np.array]:
    """Evaluate the model.

    Args:
        model: Model to evaluate.
        loader: DataLoader to use for evaluation.
        criterion: Loss function.
        device: Device to use for evaluation (default is "cpu").
        verbose: Option to print evaluation progress (default is True).

    Returns:
        Dictionary containing evaluation logs.
    """
    model.eval()

    losses = []
    scores = []

    for x_batch, y_true in tqdm(loader, desc="evaluation", file=sys.stdout, ncols=100, disable=not verbose):
        x_batch = x_batch.to(device)  # Long tensor [B, T]
        y_true = y_true.to(device)    # Long tensor [B, T]

        log_prob = model(x_batch)

        B, T = y_true.shape
        loss = criterion(log_prob.view(B * T, -1), y_true.view(B * T))

        losses.append(loss.item())

        y_pred = log_prob.argmax(2).detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        padding_mask = y_true != NER_PAD_ID
        for i in range(x_batch.size(0)):
            scores.append(sequence_f1(y_true[i][padding_mask[i]], y_pred[i][padding_mask[i]]))

    logs = {
        "losses": np.array(losses),
        "f1": np.array(scores)
    }
    return logs


# Define the LSTM architecture for NER
class NER_LSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_hidden_layers, num_classes):
        super(NER_LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size, padding_idx=TOK_PAD_ID)  # Mapping from token_id to its vector representation
        self.rnn = nn.LSTM(input_size, hidden_size, num_hidden_layers, bidirectional=True, dropout=0.2, batch_first=True)  # LSTM layer
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # Layer normalization
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Classification head

    def forward(self, x):
        x = self.embed(x)
        x, (_, _) = self.rnn(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.fc(x)
        scores = torch.log_softmax(x, dim=2)
        return scores


# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device - {device}")

# Collect unique NER tags from the dataset
targets = set()
for split in ("train", "validation", "test"):
    for sample in dataset[split]:
        targets.update(sample["ner_tags"])

# Sort the unique NER tags
targets = sorted(targets)

# Create NER datasets for training and validation
train_dataset = NERDataset(dataset["train"], vocab)
validation_dataset = NERDataset(dataset["validation"], vocab)

# Set batch size
batch_size = 64

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=ner_collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=ner_collate_fn)


# Initialize the NER LSTM model
model_lstm = NER_LSTM(len(vocab), 512, 512, 3, len(targets))
model_lstm = model_lstm.to(device)  # Move the model to the specified device

# Print the model architecture and the number of trainable parameters
print(model_lstm)
print("Number of trainable parameters -", sum(p.numel() for p in model_lstm.parameters() if p.requires_grad))

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Initialize the optimizer
optimizer = optim.Adam(model_lstm.parameters(), lr=1e-3)

# Set up a learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=.5)

# Train and validate the model
n_epochs = 12

train_losses = []
train_scores = []
valid_losses = []
valid_scores = []
best_score = float("-inf")

for ep in range(n_epochs):
    print(f"\nEpoch {ep + 1:2d}/{n_epochs:2d}")

    train_logs = train_one_epoch(model_lstm, train_loader, criterion, optimizer, device, verbose=True)
    train_losses.append(np.mean(train_logs["losses"]))
    train_scores.append(np.mean(train_logs["f1"], 0))
    print("      loss:", train_losses[-1])
    print("        f1:", train_scores[-1])

    valid_logs = evaluate(model_lstm, validation_loader, criterion, device, verbose=True)
    valid_losses.append(np.mean(valid_logs["losses"]))
    valid_scores.append(np.mean(valid_logs["f1"], 0))
    print("      loss:", valid_losses[-1])
    print("        f1:", valid_scores[-1])

    # Save the best model
    if valid_scores[-1].mean() >= best_score:
        checkpoint = {
            "model_state_dict": model_lstm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": ep,
            "num_epochs": n_epochs,
            "metrics": {
                "training": {"loss": train_losses[-1], "accuracy": train_scores[-1]},
                "validation": {"loss": valid_losses[-1], "accuracy": valid_scores[-1]},
            },
        }
        torch.save(checkpoint, "best_lstm.pth")
        print("🟢 Saved new best state! 🟢", valid_scores[-1].mean())
        best_score = valid_scores[-1].mean()  # Update best score to a new one

    scheduler.step(valid_scores[-1].mean())

print("Finished")