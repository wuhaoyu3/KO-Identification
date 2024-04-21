import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import h5py
import torch

class ProteinEmbDataset(Dataset):
    def __init__(self, archive, data='data', label='label'):
        self.archive = h5py.File(archive, 'r')
        self.data = self.archive[data]
        self.labels = self.archive[label]

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return data, label

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.archive.close()

class ProteinSeqDataset(ProteinEmbDataset):
    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return data, label


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class BinaryLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=100, vocab_size=21, dropout_prob=0.1):
        super(BinaryLSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 1D Conv layer
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=16, stride=1)

        # 1D max pooling layer
        self.max_pool1d = nn.MaxPool1d(kernel_size=5, stride=5)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=False)

        # Dense layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # 1D Conv layer
        conv_out = self.conv1d(embedded.permute(0, 2, 1))
        conv_out = torch.relu(conv_out)

        # 1D max pooling layer
        pool_out = self.max_pool1d(conv_out)

        # Dropout layer
        pool_out = self.dropout(pool_out)

        # LSTM layer
        lstm_out, _ = self.lstm(pool_out.permute(0, 2, 1))

        # Dense layer
        dense_out = self.fc(lstm_out[:, -1, :])
        dense_out = self.sigmoid(dense_out)

        return dense_out


class BinaryAttentionClassifier(nn.Module):
    def __init__(self, embedding_dim=128, vocab_size=21, dropout_prob=0.1):
        super(BinaryAttentionClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 1D Conv layer
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=16, stride=1)

        # 1D max pooling layer
        self.max_pool1d = nn.MaxPool1d(kernel_size=5, stride=5)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1)

        # Dense layer
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding layer
        # print(f'input shape: {x.shape}')
        embedded = self.embedding(x)
        # print(f'embed shape: {embedded.shape}')

        # 1D Conv layer
        conv_out = self.conv1d(embedded.permute(0, 2, 1))
        # print(f'conv1d shape: {conv_out.shape}')
        conv_out = torch.relu(conv_out)
        # print(f'relu shape: {conv_out.shape}')

        # 1D max pooling layer
        pool_out = self.max_pool1d(conv_out)
        # print(f'pool shape: {pool_out.shape}')

        # Dropout layer
        pool_out = self.dropout(pool_out)

        # ATT layer
        x = pool_out.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        att_out = x.permute(1, 2, 0)

        # Dense layer
        dense_out = self.fc(att_out[:, :, -1])
        # print(f'fc shape: {dense_out.shape}')
        dense_out = self.sigmoid(dense_out)
        # print(f'sigmoid shape: {dense_out.shape}')

        return dense_out


def train_model(model, num_epochs, train_loader, valid_loader, optimizer, criterion, patience, device, model_save_path):
    best_valid_loss = float('inf')
    counter = 0
    best_models_loss = []

    model.to(device)
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        batches = len(train_loader)
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss

            # Output the current progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"\rEpoch {epoch + 1}, Batch {i + 1}/{batches}, Train Loss: {train_loss / (i + 1)}", end='',
                      flush=True)

        # Validation loop
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                valid_loss += criterion(y_pred, y_batch)
            valid_loss /= len(valid_loader)

        # Print the training and validation loss after each epoch
        print(f"\nEpoch {epoch + 1}: Train Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss}")

        # Check if the validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0

            # Save best three models
            save_path = os.path.join(model_save_path, 'model')
            model_num = len(best_models_loss)
            if model_num < 3:
                best_models_loss.append(valid_loss)
                torch.save(model.state_dict(), save_path + str(model_num) + '.pt')
            else:
                max_loss_index = best_models_loss.index(max(best_models_loss))
                torch.save(model.state_dict(), save_path + str(max_loss_index) + '.pt')
                best_models_loss[max_loss_index] = valid_loss
        else:
            counter += 1
            if counter >= patience:
                print(f"Validation loss did not improve for {patience} epochs. Training stopped.")
                break


def test_model(model, test_loader, device):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    with torch.no_grad():
        batches = len(test_loader)
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.reshape(-1, 1).to(device)

            outputs = model(x_batch)
            total_correct += (outputs.round() == y_batch).sum().item()
            total_samples += x_batch.size(0)
            outputs = outputs.round()
            total_true_positives += ((outputs == y_batch) & (outputs == 1)).sum().item()
            total_false_positives += ((outputs != y_batch) & (outputs == 1)).sum().item()
            total_false_negatives += ((outputs != y_batch) & (outputs == 0)).sum().item()

            if (i + 1) % 10 == 0:
                print(f"\rBatch {i + 1}/{batches}", end='', flush=True)

    test_acc = total_correct / total_samples
    test_precision = total_true_positives / (total_true_positives + total_false_positives)
    test_recall = total_true_positives / (total_true_positives + total_false_negatives)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    print("\nTest Accuracy: {:.4f}, Test Precision: {:.4f}, Test Recall: {:.4f}, Test F1: {:.4f}".format(test_acc, test_precision, test_recall, test_f1))


def split_dataset(train_data, batch_size):
    valid_ratio = 0.9
    n_train_examples = int(len(train_data) * valid_ratio)
    n_valid_examples = len(train_data) - n_train_examples
    train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader
