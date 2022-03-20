# importing dependencies
from torch import nn
import torch
from DataHandling import Kitti_DataLoader
from torchsummary import summary


def condensing_layer(input_dim, output_dim, dropout=0.25):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )


def expanding_layer(input_dim,output_dim, dropout=0.25):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode="bilinear", align_corners=False),
        nn.Conv2d(input_dim, output_dim,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU()
    )


class Generator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            condensing_layer(input_dim=im_dim,output_dim=hidden_dim),
            condensing_layer(input_dim=hidden_dim, output_dim=hidden_dim*2),
            condensing_layer(input_dim=hidden_dim*2, output_dim=hidden_dim * 4),

            condensing_layer(input_dim=hidden_dim*4,output_dim=hidden_dim*8),
            expanding_layer(input_dim=hidden_dim*8, output_dim=hidden_dim*4),
            expanding_layer(input_dim=hidden_dim*4,output_dim=hidden_dim*2),
            expanding_layer(input_dim=hidden_dim*2,output_dim=hidden_dim),
            expanding_layer(input_dim=hidden_dim,output_dim=im_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4, input_shape=(256, 1024)):
        super().__init__()

        self.model = nn.Sequential(
            condensing_layer(input_dim=im_dim, output_dim=hidden_dim),  # input shape halves
            condensing_layer(input_dim=hidden_dim, output_dim=hidden_dim * 2),  # input shape halves
            condensing_layer(input_dim=hidden_dim * 2, output_dim=hidden_dim * 4),  # input shape halves
            condensing_layer(input_dim=hidden_dim * 4, output_dim=hidden_dim * 8),  # input shape halves
            condensing_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 16),  # input shape halves
            nn.Flatten(),
            nn.Linear(in_features=input_shape[0]*input_shape[1]//2**4, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
            )


    def forward(self, x: torch.Tensor):
        return self.model(x)

