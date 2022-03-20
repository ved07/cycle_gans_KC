# importing dependencies
from torch import nn
import torch
from DataHandling import Kitti_DataLoader
from torchsummary import summary



def conv_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1),  # input shape - 2
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2)
    )


def conv_transpose_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=1), # input shape + 6
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2)
    )


class Generator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            conv_layer(im_dim, hidden_dim),
            conv_layer(hidden_dim, hidden_dim*2),
            conv_layer(hidden_dim*2,hidden_dim*4),
            conv_transpose_layer(hidden_dim*4, hidden_dim*2),
            conv_transpose_layer(hidden_dim*2, hidden_dim),
            conv_transpose_layer(hidden_dim, im_dim))


    def forward(self, x: torch.Tensor):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4, input_shape=(256, 256)):
        super().__init__()

        self.model = nn.Sequential(
            conv_layer(im_dim, hidden_dim), # -6
            conv_layer(hidden_dim, hidden_dim*2), # -6
            conv_layer(hidden_dim*2, hidden_dim*4), # -6
            nn.Flatten(),
            nn.Linear(in_features=(input_shape[0]-6)*(input_shape[1]-6)*hidden_dim*4, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
            )


    def forward(self, x: torch.Tensor):
        return self.model(x)

