from torch import nn
import numpy as np
class My_Model(nn.Module) :
    def __int__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Sigmoid(),
            nn.Linear(100, 30),
            nn.Sigmoid(),
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        return np.argmax(x)
