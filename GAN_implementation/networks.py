# importing dependencies
from torch import nn
import torch


def conv_transpose_layer(input_dim, output_dim, filter_size = (2, 2), stride = (2, 2), padding=(0, 0)):
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_size=filter_size, stride=stride, padding=padding),
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2)
    )


def condensing_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=(2, 2), stride=(2, 2)),
        nn.LeakyReLU(0.2)
        )

# shape is 256 x 1024
class Generator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            conv_transpose_layer(input_dim=64, output_dim=64),
            # shape is [?, 32, 32 ,128]
            conv_transpose_layer(input_dim=64, output_dim=32),
            # shape is [?, 16, 64 ,256]
            conv_transpose_layer(input_dim=32, output_dim=32),
            # shape is [?, 8, 128 ,512]
            conv_transpose_layer(input_dim=32, output_dim=16),
            # shape is [?, 4, 256 ,1024]
            conv_transpose_layer(input_dim=16, output_dim=16),
            # shape is [?, 16, 512 ,2048]
            conv_transpose_layer(input_dim=16, output_dim=8),
            # shape is [?, 8, 1024 ,4096]
            condensing_layer(input_dim=8, output_dim=4),
            # shape is [?, 8, 512 ,2048]
            condensing_layer(input_dim=4, output_dim=3),
            # shape is [?, 3, 256 ,1024]
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
            nn.Flatten(),
            nn.Linear(in_features=input_shape[0]*input_shape[1]//8, out_features=1),
            nn.Sigmoid()
            )


    def forward(self, x:torch.Tensor):
        return self.model(x)


def generate_noise(batch_size=1, input_shape=64, l = 16, r = 64):
    vec = torch.rand(input_shape*batch_size * l * r)
    vec = torch.reshape(vec, [batch_size, input_shape, l, r])
    return vec


