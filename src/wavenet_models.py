# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.nn.functional as F


class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, activation='relu', padding=1, kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels

        if activation == 'relu':
            self.activation = lambda *args, **kwargs: F.relu(*args, **kwargs, inplace=True)
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'glu':
            self.activation = F.glu
            in_channels = channels // 2

        self.left_pad = left_pad
        self.dilated_conv = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=1,
                                      padding=dilation * padding, dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels,
                                  kernel_size=1, bias=True)

    def forward(self, input):
        x = input

        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        x = self.dilated_conv(x)
        x = self.activation(x)
        x = self.conv_1x1(x)

        return input + x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_blocks = args.encoder_blocks
        self.n_layers = args.encoder_layers
        self.channels = args.encoder_channels
        self.latent_channels = args.latent_d
        self.activation = args.encoder_func

        try:
            self.encoder_pool = args.encoder_pool
        except AttributeError:
            self.encoder_pool = 800

        layers = []
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                layers.append(DilatedResConv(self.channels, dilation, self.activation))
        self.dilated_convs = nn.Sequential(*layers)

        self.start = nn.Conv1d(1, self.channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv_1x1 = nn.Conv1d(self.channels, self.latent_channels, 1)
        self.pool = nn.AvgPool1d(self.encoder_pool)

    def forward(self, x):
        x = x / 255 - .5
        if x.dim() < 3:
            x = x.unsqueeze(1)

        x = self.start(x)
        x = self.dilated_convs(x)
        x = self.conv_1x1(x)
        x = self.pool(x)

        return x


class ZDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_classes = args.n_datasets

        convs = []
        for i in range(args.d_layers):
            in_channels = args.latent_d if i == 0 else args.d_channels
            convs.append(nn.Conv1d(in_channels, args.d_channels, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(args.d_channels, self.n_classes, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=args.p_dropout_discriminator)

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)

        mean = logits.mean(2)
        return mean


def cross_entropy_loss(input, target):
    # input:  (batch, 256, len)
    # target: (batch, len)

    batch, channel, seq = input.size()

    input = input.transpose(1, 2).contiguous()
    input = input.view(-1, 256)  # (batch * seq, 256)
    target = target.view(-1).long()  # (batch * seq)

    cross_entropy = F.cross_entropy(input, target, reduction='none')  # (batch * seq)
    return cross_entropy.reshape(batch, seq).mean(dim=1)  # (batch)
