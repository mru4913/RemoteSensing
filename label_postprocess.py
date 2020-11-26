
import cv2 
import torch 
import numpy as np

from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader

from utils.data import myDataset
# <--- import your net 

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

def count_label(counter, pred, true, label_value):
    true = true.numpy()
    mask = pred == label_value 
    values = true[mask]
    return Counter(values) + counter

def load_state(model, device, state_dict_file):
    with open(state_dict_file, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model 

def predict(model, test_loader, device):
    print('Start evaluating...')
    # set model to evaluate model
    model.eval()
    # deactivate autograd engine and 
    # reduce memory usage and speed up computations

    label4_counter = Counter()
    label5_counter = Counter()
    label6_counter = Counter()
    label8_counter = Counter()

    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1]

            outputs = model(*inputs)
            outputs = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8) + 1 

            label4_counter = count_label(label4_counter, outputs, labels, 4)
            label5_counter = count_label(label5_counter, outputs, labels, 5)
            label6_counter = count_label(label6_counter, outputs, labels, 6)
            label8_counter = count_label(label8_counter, outputs, labels, 8)

    print("label4_counter: ",label4_counter)
    print("label5_counter: ",label5_counter)
    print("label6_counter: ",label6_counter)
    print("label8_counter: ",label8_counter)


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = "xxx.txt"
    state_dict_file =  "xxx.pth"
    model = net()
    model = load_state(model, device, state_dict_file) 

    test = myDataset(file_path, transforms=False) 
    test_loader = DataLoader(test, batch_size=64, num_workers=8)

    predict(model, test_loader, device)

