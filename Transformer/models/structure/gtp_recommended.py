import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# 1-st layer transformer
class FirstLayerTransformer(nn.Module):
    def __init__(self):
        super(FirstLayerTransformer, self).__init__()