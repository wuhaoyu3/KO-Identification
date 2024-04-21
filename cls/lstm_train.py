import torch
import binary_classifier
import torch.nn as nn
from torch.utils.data import ConcatDataset

gpu = 'cuda:0'
num_epochs = 1000
learning_rate = 0.001
batch_size = 200
patience = 10
model_save_path = '../../model'
train_file = ['train_dataset1.h5', 'train_dataset2.h5']

# Loading data
datasets = []
for file_path in train_file:
    dataset = binary_classifier.ProteinSeqDataset(file_path, data='data', label='label')
    datasets.append(dataset)
train_data = ConcatDataset(datasets)

# split dataset
train_dataloader, val_dataloader = binary_classifier.split_dataset(train_data, batch_size)

# Instantiate the model
model = binary_classifier.BinaryLSTMClassifier(embedding_dim=128, hidden_dim=100, vocab_size=21, dropout_prob=0.1)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=0)

# Training loop
device = torch.device(gpu)
binary_classifier.train_model(model, num_epochs, train_dataloader, val_dataloader, optimizer, criterion, patience, device, model_save_path)
