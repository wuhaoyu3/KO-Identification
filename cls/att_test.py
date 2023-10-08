import torch
import binary_classifier
import torch.nn as nn
from torch.utils.data import DataLoader

gpu = 'cuda:0'
batch_size = 200
model_path = '../../model/att_cls.pt'
test_file = ['test_dataset1.h5', 'test_dataset2.h5']

datasets = []
for file_path in test_file:
    print(file_path)
    dataset = binary_classifier.ProteinSeqDataset(file_path, data='data', label='label')
    datasets.append(dataset)
test_data = ConcatDataset(datasets)
test_loader = DataLoader(test_data, shuffle=False, num_workers=15, batch_size=batch_size)

# Instantiate the model
device = torch.device(gpu)
model = binary_classifier.BinaryAttentionClassifier(embedding_dim=128, vocab_size=21, dropout_prob=0.1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
binary_classifier.test_model(model, test_loader, device)
