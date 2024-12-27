# model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt  # Importing Matplotlib

# Configure logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


class NeuMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, mf_dim: int = 8, layers: List[int] = [64, 32, 16, 8],
                 dropout: float = 0.2):
        super(NeuMF, self).__init__()
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, mf_dim)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding_mlp = nn.Embedding(num_items, layers[0] // 2)

        # Initialize MLP layers
        mlp_layers = []
        input_dim = layers[0]
        for layer_size in layers[1:]:
            mlp_layers.append(nn.Linear(input_dim, layer_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_dim = layer_size
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.final = nn.Linear(mf_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.final.weight)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # GMF part
        gmf_user = self.user_embedding_gmf(user_indices)
        gmf_item = self.item_embedding_gmf(item_indices)
        gmf_out = gmf_user * gmf_item  # Element-wise product

        # MLP part
        mlp_user = self.user_embedding_mlp(user_indices)
        mlp_item = self.item_embedding_mlp(item_indices)
        mlp_cat = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_cat)

        # Concatenate GMF and MLP parts
        neu_out = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.final(neu_out)
        return self.sigmoid(score)


def plot_metrics(history: dict):
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(18, 12))

    # Plot Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['loss'], 'r-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['accuracy'], 'b-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['precision'], 'g-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['recall'], 'c-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Plot F1-Score
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['f1'], 'm-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()

    # Plot ROC AUC
    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['roc_auc'], 'y-', label='ROC AUC')
    plt.title('ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')  # Save the figure
    plt.close()  # Close the figure to free memory


def train_neumf_model(
        data: List[Tuple[int, int, float]],
        num_users: int,
        num_items: int,
        epochs: int = 20,
        lr: float = 0.001,
        neg_per_pos: int = 4,
        batch_size: int = 256,
        dropout: float = 0.2
) -> NeuMF:
    """
    Trains the NeuMF model on the provided data.

    Args:
        data: List of tuples containing (user_index, item_index, rating).
        num_users: Total number of unique users.
        num_items: Total number of unique items.
        epochs: Number of training epochs.
        lr: Learning rate.
        neg_per_pos: Number of negative samples per positive sample.
        batch_size: Training batch size.
        dropout: Dropout rate in MLP layers.

    Returns:
        Trained NeuMF model.
    """
    # Separate positive interactions
    positive_data = [(u, i) for (u, i, r) in data if r > 0]
    positive_set = set(positive_data)

    # Generate training samples with negative sampling
    train_samples = []
    for (u, i) in positive_data:
        train_samples.append((u, i, 1))
        for _ in range(neg_per_pos):
            neg_i = np.random.randint(num_items)
            while (u, neg_i) in positive_set:
                neg_i = np.random.randint(num_items)
            train_samples.append((u, neg_i, 0))

    # Shuffle the training samples
    np.random.shuffle(train_samples)

    # Convert to tensors
    users = torch.tensor([x[0] for x in train_samples], dtype=torch.long)
    items = torch.tensor([x[1] for x in train_samples], dtype=torch.long)
    labels = torch.tensor([x[2] for x in train_samples], dtype=torch.float32)

    # Initialize model, loss function, and optimizer
    model = NeuMF(num_users, num_items, dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize lists to store metrics
    history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }

    # Training loop
    for epoch in range(epochs):
        permutation = torch.randperm(len(users))
        total_loss = 0.0
        model.train()

        # Lists to accumulate true labels and predictions for metrics
        all_labels = []
        all_preds = []

        for i in range(0, len(users), batch_size):
            idx = permutation[i:i + batch_size]
            u_batch, i_batch, l_batch = users[idx], items[idx], labels[idx]

            optimizer.zero_grad()
            preds = model(u_batch, i_batch).squeeze()
            loss = criterion(preds, l_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(l_batch)

            # Accumulate labels and predictions
            all_labels.extend(l_batch.tolist())
            all_preds.extend(preds.tolist())

        avg_loss = total_loss / len(users)

        # Convert predictions to binary (threshold=0.5)
        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

        # Calculate metrics
        accuracy = accuracy_score(all_labels, binary_preds)
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            roc_auc = 0.0  # Handle cases where ROC AUC cannot be computed

        # Append metrics to history
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        history['roc_auc'].append(roc_auc)

        # Log and print metrics
        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
        )
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
        )

    # Conclusion: Log and print final metrics
    logging.info("Training completed.")
    print("Training completed.")

    # Plot metrics
    plot_metrics(history)

    return model
