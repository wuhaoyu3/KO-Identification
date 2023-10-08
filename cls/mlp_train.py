import torch
import torch.nn as nn
from torch.utils.data import DataLoader,  ConcatDataset
from torch.utils import data
import random
import numpy as np
import binary_classifier

gpu = 'cuda:0'
input_size = 1024
hidden_size = 100
output_size = 1
num_epochs = 1000
learning_rate = 0.001
batch_size = 200
patience = 10
model_save_path = '../../model'
train_file = ['train_dataset1.h5', 'train_dataset2.h5']

# Loading data
datasets = []
for file_path in train_file:
    dataset = binary_classifier.ProteinEmbDataset(file_path, data='data', label='label')
    datasets.append(dataset)
train_data = ConcatDataset(datasets)

# split train and valid
train_dataloader, val_dataloader = binary_classifier.split_dataset(train_data, batch_size)

# Define the MLP classifier
device = torch.device(gpu)
model = binary_classifier.MLPClassifier(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the MLP classifier
binary_classifier.train_model(model, num_epochs, train_dataloader, valid_dataloader, optimizer, criterion, patience, device, model_save_path)
