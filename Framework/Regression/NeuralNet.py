"""
Regression using Neural Networks (ISBI)

Examples:
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
    https://github.com/OValery16/Tutorial-about-3D-convolutional-network/blob/master/model.py
    https://discuss.pytorch.org/t/multiple-input-model-architecture/19754/2

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, num_features=None, num_hidden_neurons=None, num_output=None, n_hidden_layers=1, drop_prob=None):
        super(SimpleNN, self).__init__()

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        hidden_block = nn.ModuleList()
        for _ in range(n_hidden_layers):
            hidden_layer = HiddenBlock(num_features,
                                       num_hidden_neurons,
                                       drop_prob)
            hidden_block.append(hidden_layer)
            num_features = num_hidden_neurons
        self.hidden_block = nn.Sequential(*hidden_block)
        self.output = nn.Linear(num_hidden_neurons, num_output)
        # nn.init.normal_(self.output.weight, mean=0, std=0.1)
        # nn.init.constant_(self.output.bias, val=0)

    def forward(self, x):
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden_block(x)
        x = self.output(x)

        return x


class HiddenBlock(nn.Module):
    def __init__(self, in_features, out_features, drop_prob):
        super().__init__()
        self.hidden = nn.Linear(in_features, out_features)
        # nn.init.normal_(self.hidden.weight, mean=0, std=0.1)
        # nn.init.constant_(self.hidden.bias, val=0)

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


"""
Simple CNN architecture for lu/vector prediction
"""
class SimpleNet3D(nn.Module):
    def __init__(self, window=None, channels=1):
        super(SimpleNet3D, self).__init__()

        # features (direction)
        self.features = nn.Linear(3, 10)
        self.relu = nn.ReLU()

        # window
        self.window = window

        # multiple input model architecture (window + direction)
        if window == 9:
            # input image:  1@9x9x9
            self.conv_layer1 = self._make_conv_layer(channels, 16)     # 16@3x3x3
            self.fc1 = nn.Linear(16 * 3**3 + 10, 32)           # 16 * 3**3 + 10, 32

        elif window == 11:
            # input image:  1@11x11x11
            self.conv_layer1 = self._make_conv_layer(channels, 16)     # 16@7x7x7
            self.conv_layer2 = self._make_conv_layer(16, 32)    # 32@3x3x3
            self.fc1 = nn.Linear(32 * 3**3 + 10, 128)                # 864->128

        self.fc2 = nn.Linear(32, 8)   # 32, 8
        self.fc3 = nn.Linear(8, 3)     # 64, 3
        self.leakyrelu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(32)   #32
        self.batch2 = nn.BatchNorm1d(8)    #8
        self.drop = nn.Dropout(p=0.1)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def _make_conv_layer_2(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def forward(self, xd, xw):
        # x (window), y(direction)
        xd = self.features(xd)
        # print('     features(xd)={}'.format(xd.size()))
        xd = self.relu(xd)
        # print('     relu(features(xd))={}'.format(xd.size()))

        if self.window == 9:
            xw = self.conv_layer1(xw)
            # print('     conv_layer1(xw)={}'.format(xw.size()))

        elif self.window == 11:
            xw = self.conv_layer1(xw)
            xw = self.conv_layer2(xw)

        xd = xd.view(xd.size(0), -1)
        # print('     view(xd)={}'.format(xd.size()))
        xw = xw.view(xw.size(0), -1)
        # print('     view(xw)={}'.format(xw.size()))

        # multiple input model architecture (window + direction)
        x = torch.cat((xd,xw), dim=1)
        # print('     cat(xd,xw)={}'.format(x.size()))

        x = self.fc1(x)
        x = self.leakyrelu(x)
        # print('     leakyrelu(x)={}'.format(x.size()))
        self.eval()
        x = self.batch1(x)
        self.train()
        x = self.drop(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)
        self.eval()
        x = self.batch2(x)
        self.train()
        x = self.drop(x)

        x = self.fc3(x)

        return x


"""
Complex CNN architecture for gu prediction
"""
class ComplexNet3D(nn.Module):
    def __init__(self, window=None, channels=1):
        super(ComplexNet3D, self).__init__()

        # features (direction+plan+impl)
        self.features = nn.Linear(9, 100)  #12,100
        self.relu = nn.ReLU()

        # window
        self.window = window

        # multiple input model architecture (window + direction)
        if window == 9:
            # input image:  1@9x9x9
            self.conv_layer1 = self._make_conv_layer(channels, 16)     # 16@3x3x3
            self.fc1 = nn.Linear(16 * 3**3 + 100, 128)           # 16 * 3**3 + 10, 32

        self.fc2 = nn.Linear(128, 16)   # 32, 8
        self.fc3 = nn.Linear(16, 3)     # 64, 3
        self.leakyrelu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(128)   #32
        self.batch2 = nn.BatchNorm1d(16)    #8
        self.drop = nn.Dropout(p=0.1)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def forward(self, xd, xp, xi, xg, xw):
        # concat inputs: xd[n,3], xp[n,3], xi[n,3], xg[n,3] into xi[n,9]
        xi = torch.cat((xd,xp,xi), dim=1)
        # xi = torch.cat((xd, xp, xi, xg), dim=1)

        # x (window), y(direction)
        xi = self.features(xi)  # xi
        # print('     features(xd)={}'.format(xd.size()))
        xi = self.relu(xi)
        # print('     relu(features(xd))={}'.format(xd.size()))

        if self.window == 9:
            xw = self.conv_layer1(xw)
            # print('     conv_layer1(xw)={}'.format(xw.size()))

        xi = xi.view(xi.size(0), -1)
        # print('     view(xd)={}'.format(xd.size()))
        xw = xw.view(xw.size(0), -1)
        # print('     view(xw)={}'.format(xw.size()))

        # multiple input model architecture (window + direction)
        x = torch.cat((xi,xw), dim=1)
        # print('     cat(xd,xw)={}'.format(x.size()))

        x = self.fc1(x)
        x = self.leakyrelu(x)
        # print('     leakyrelu(x)={}'.format(x.size()))
        self.eval()
        x = self.batch1(x)
        self.train()
        x = self.drop(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)
        self.eval()
        x = self.batch2(x)
        self.train()
        x = self.drop(x)

        x = self.fc3(x)

        return x