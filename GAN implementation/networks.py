# importing dependencies
from torch import nn
import torch
from DataHandling import Kitti_DataLoader


def conv_transpose_layer(input_dim, output_dim, filter_size, stride, padding=(0, 0)):
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_size=filter_size, stride=stride, padding=padding),
        nn.BatchNorm2d(output_dim),
        nn.ReLU()
    )


# shape is 256 x 1024
class Generator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            conv_transpose_layer(input_dim=input_dim, output_dim=128, filter_size=(16, 16), stride=(1, 1)),
            # shape is [?, 64, 16 ,16]
            conv_transpose_layer(input_dim=128, output_dim=64, filter_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # shape is [?, 32, 32 ,32]
            conv_transpose_layer(input_dim=64, output_dim=32, filter_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # shape is [?, 16, 64 ,64]
            conv_transpose_layer(input_dim=32, output_dim=16, filter_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # shape is [?, 8, 128 ,128]
            conv_transpose_layer(input_dim=16, output_dim=4, filter_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # shape is [?, 4, 256 ,256]
            conv_transpose_layer(input_dim=4, output_dim=3, filter_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            # shape is [?, 3, 512 ,256]
            conv_transpose_layer(input_dim=3, output_dim=3, filter_size=(1, 4), stride=(1, 2), padding=(0, 1))
            # shape is [?, 3, 1024 ,256]
        )


    def forward(self, x: torch.Tensor):
        return self.model(x)


def condensing_layer(input_dim, output_dim, dropout=0.25):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.ReLU()
        )


class Discriminator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4, input_shape=(256, 1024)):
        super().__init__()

        self.model = nn.Sequential(
            condensing_layer(input_dim=im_dim, output_dim=hidden_dim),  # input shape halves
            condensing_layer(input_dim=hidden_dim, output_dim=hidden_dim * 2),  # input shape halves
            condensing_layer(input_dim=hidden_dim * 2, output_dim=hidden_dim * 8),  # input shape halves
            condensing_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 16),  # input shape halves
            nn.Flatten(),
            nn.Linear(in_features=input_shape[0]*input_shape[1]//2**4, out_features=1),
            nn.Sigmoid()
            )


    def forward(self, x:torch.Tensor):
        return self.model(x)


def generate_noise(batch_size=1, input_shape=64):
    vec = torch.rand(input_shape*batch_size)
    vec = torch.reshape(vec, [batch_size, input_shape, 1, 1])
    return vec


