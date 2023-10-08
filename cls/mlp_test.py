import torch
from torch.utils.data import DataLoader, ConcatDataset
import binary_classifier
import os

gpu = 'cuda:0'
input_size = 1024
hidden_size = 100
output_size = 1
batch_size = 200
model_path = '../../model/mlp_cls.pt'
test_file = ['test_dataset1.h5', 'test_dataset2.h5']

datasets = []
for file_path in test_file:
    print(file_path)
    dataset = binary_classifier.ProteinEmbDataset(file_path, data='data', label='label')
    datasets.append(dataset)
test_data = ConcatDataset(datasets)

print(f'Number of testing examples: {len(test_data)}')

test_loader = DataLoader(test_data, shuffle=False, num_workers=15, batch_size=batch_size)

# load the model
device = torch.device(gpu)
model = binary_classifier.MLPClassifier(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

binary_classifier.test_model(model, test_loader, device)
