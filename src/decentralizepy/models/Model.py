from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.accumulated_gradients = []