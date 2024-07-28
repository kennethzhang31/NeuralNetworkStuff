import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Encoder, self).__init__()
        self.ch = ch
        # (3, 64, 64)
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(ch)
        # (ch, 32, 32)
        self.conv2 = Conv2d(ch, ch*2, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(ch*2)
        # TODO: Finish the model.
        # (ch*2, 16, 16)
        self.conv3 = Conv2d(ch*2, ch*4, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(ch*4)
        # (ch*4, 8, 8)
        self.conv4 = Conv2d(ch*4, ch*8, 5, stride=2)
        self.bn4 = nn.BatchNorm2d(ch*8)
        # (ch*8, 4, 4)
        self.fc_mean = nn.Linear(ch*8*4*4, z_dim)
        self.fc_logvar = nn.Linear(ch*8*4*4, z_dim)
        # (z_dim)

    def forward(self, x):
        # TODO: Finish the forward propagation.
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), inplace=True)
        x = x.view(x.size(0), -1)
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

class Generator(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Generator, self).__init__()
        self.ch = ch
        self.fc1 = nn.Linear(z_dim, 8*8*ch*8)
        # (ch*8, 8, 8)
        self.conv1 = Conv2d(ch*8, ch*4, 3)
        self.bn1 = nn.BatchNorm2d(ch*4)
        # (ch*4, 8, 8)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(ch*4, ch*2, 3)
        self.bn2 = nn.BatchNorm2d(ch*2)
        # TODO: Finish the model.
        # (ch*2, 16, 16)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(ch*2, ch, 3)
        self.bn3 = nn.BatchNorm2d(ch)
        # (ch, 32, 32)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(ch, 3, 3)
        # (3, 64, 64)

    def forward(self, z):
        # TODO: Finish the forward propagation.
        x = F.leaky_relu(self.fc1(z), inplace=True)
        x = x.view(-1, self.ch*8,8,8)

        x = self.up2(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.up3(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.up4(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        x = torch.tanh(self.conv4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super(Discriminator, self).__init__()
        self.ch = ch
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        self.conv2 = nn.utils.spectral_norm(Conv2d(ch, ch*2, 5, stride=2))
        # TODO: Finish the model.
        self.bn2 = nn.BatchNorm2d(ch*2)
        self.conv3 = nn.utils.spectral_norm(Conv2d(ch*2, ch*4, 5, stride=2))
        self.bn3 = nn.BatchNorm2d(ch*4)
        self.conv4 = nn.utils.spectral_norm(Conv2d(ch*4, ch*8, 5, stride=2))
        self.bn4 = nn.BatchNorm2d(ch*8)
        self.fc = nn.Linear(ch*8*4*4, 1)

    def forward(self, x):
        # TODO: Finish the forward propagation.
        x = F.leaky_relu(self.conv1(x), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), inplace=True)
        x = x.view(x.size(0), -1)
        d_logit = self.fc(x)
        d_prob = torch.sigmoid(d_logit)
        return d_prob, d_logit
