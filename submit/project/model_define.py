import os 
import torch 
import torch.nn as nn
# import numpy as np 

################################################################################
#### ADD MODEL ARC HERE ***
### e.g. from .layer import doubleConv 
### e.g. from .loader import SegDataset  
################################################################################

# e.g. from .loader import SegDataset  



def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # change the model name 
    path = os.path.join(os.path.dirname(__file__), 'xxx.pth')

    # define your model here and all the necessary hyperparameters
    model = net() # <- here 
    model.to(device)

    # load model
    with open(path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # evaluation mode 
    model.eval()
    return model
