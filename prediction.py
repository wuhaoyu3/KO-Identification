# python pre.py model pre out
import h5py
import sys
import torch
import binary_classifier

BATCH_SIZE = 200
gpu = 'cuda:0'
input_size = 1024
hidden_size = 100
output_size = 1

model_path = sys.argv[1]
pre_file = sys.argv[2]
out_file = sys.argv[3]


# load the model
device = torch.device(gpu)
model = binary_classifier.MLPClassifier(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load(model_path))

# set the model to evaluation mode
model.eval()

fo = open(out_file, 'w')
with h5py.File(pre_file, 'r') as f:
    for i, key in enumerate(f.keys()):
        identifier = key.split()[0]
        embedding = torch.tensor(f[key][:]).to(device)
        Y_preds = model(embedding).round().item()
        fo.writelines(identifier + ',' + str(int(Y_preds)) + '\n')
        print("\rreading seq {}".format(i).ljust(60), end='', flush=True)
fo.close()
