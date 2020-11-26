from .lovasz_losses import *

import torch.nn as nn
import torch.nn.functional as F

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=True, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, classes=self.classes, per_image=self.per_image, ignore=self.ignore)
        return loss