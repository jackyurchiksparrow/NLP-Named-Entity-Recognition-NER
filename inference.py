import sys  # System-specific parameters and functions
import numpy as np  # NumPy for numerical operations
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network module
from typing import List, Tuple, Mapping  # Type hinting for better code readability
import torch.nn.functional as F  # Functional operations like activation functions and loss calculations
from tqdm import tqdm  # Progress bar library for iterations
from torch.utils.data import DataLoader  # DataLoader for efficient data handling
from torch.nn.utils.rnn import pad_sequence  # Utility for padding sequences in RNNs
from torch.utils.data import Dataset, DataLoader  # Dataset and DataLoader for efficient data handling and batching
import datasets

torch.manual_seed(42)

# Define token and padding IDs
TOK_PAD_ID = 0  # Define a token ID for padding input sequences.
NER_PAD_ID = -100  # Padding ID for NER tags

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
        """
        Get the number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.samples)  # Return the number of samples in the dataset.

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
        doc = torch.LongTensor([self.vocabulary[token] for token in sample["tokens"]])  # Convert tokens to LongTensor indices.
        
        # Convert NER tags to a tensor of Long values.
        label = torch.LongTensor(sample["ner_tags"])  # Convert NER tags to LongTensor.

        return doc, label  # Return the token indices and NER tags for the sample.

# Define the NER LSTM class
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
    """Calculate F1 score for one sequence."""
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

# Load the dataset
dataset = datasets.load_dataset("benjamin/ner-uk")

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

# Collect unique NER tags from the dataset
targets = set()
for split in ("train", "validation", "test"):
    for sample in dataset[split]:
        targets.update(sample["ner_tags"])

# Sort the unique NER tags
targets = sorted(targets)

# creating a custom dataset object
test_dataset = NERDataset(dataset["test"], vocab)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device - {device}")

# Load the saved checkpoint
checkpoint = torch.load("models/best_lstm.pth")

# Create a model instance with the same architecture as the saved model
model_lstm = NER_LSTM(len(vocab), 512, 512, 3, len(targets))

# Load the model's state dictionary
model_lstm.load_state_dict(checkpoint["model_state_dict"])

model_lstm.to(device)

# Set the model to evaluation mode
model_lstm.eval()

# Initialize lists to store scores and accuracies
scores = []
accuracies = []

# Create a test data loader (you should define your test dataset similar to the training dataset)
# Assuming you have a test dataset prepared as 'test_dataset'
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=ner_collate_fn)

# Iterate over batches in the test data loader
for x_batch, y_true in tqdm(test_loader, desc="test evaluation", file=sys.stdout, ncols=100, disable=not True):
    # Move input and true label tensors to the specified device (e.g., GPU)
    x_batch = x_batch.to(device)
    y_true = y_true.to(device)

    # Get log probabilities from the model for the current batch
    log_prob = model_lstm(x_batch)

    # Get predicted class indices by taking the argmax of the log probabilities
    y_pred = log_prob.argmax(2).detach().cpu().numpy()  # Convert to numpy array
    y_true = y_true.detach().cpu().numpy()  # Convert true labels to numpy array
    
    # Create a mask to ignore padding tokens in the true labels
    padding_mask = y_true != NER_PAD_ID

    # Iterate over each sample in the batch
    for i in range(x_batch.size(0)):
        # Calculate the F1 score for the current sample and append to scores
        scores.append(np.mean(sequence_f1(y_true[i][padding_mask[i]], y_pred[i][padding_mask[i]])))

        # Calculate accuracy for the current sample
        correct_tokens = np.sum(y_true[i][padding_mask[i]] == y_pred[i][padding_mask[i]])  # Count correct predictions
        total_tokens = np.sum(padding_mask[i])  # Count total tokens excluding padding
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0  # Compute accuracy
        accuracies.append(accuracy)  # Append accuracy to the list

# Print the calculated accuracies and scores
print(f"Accuracies: {accuracies}")
print(f"Scores: {scores}")
print("Done")