import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import math

__all__ = ['PeleeNet17b', 'PeleeNet31b']


def PeleeNet17b():
    return PeleeNetV2(block_config=[3, 4])


def PeleeNet31b():
    return PeleeNetV2(
                        block_config = [3,8], growth_rate=[32,48], 
                        bottleneck_width=[4,4], compression_factor=0.5)


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bottleneck_width):
        super(_DenseLayer, self).__init__()


        growth_rate = growth_rate // 2
        inter_channel = growth_rate  * bottleneck_width  

        self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)

        return torch.cat([x, branch1, branch2], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features/2)

        self.stem1 = BasicConv2d(num_input_channels, num_init_features, kernel_size=5, stride=3, padding=2)
        self.stem2a = BasicConv2d(num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = BasicConv2d(num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = BasicConv2d(2*num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out


class _SEBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) 
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x

class PeleeNetV2(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=[32, 32, 64, 64], block_config=[3, 4, 4, 6],
                 num_init_features=32, bottleneck_width=[1, 2, 4, 4], drop_rate=0, compression_factor=1.0, num_out_features=None, use_se=True):

        super(PeleeNetV2, self).__init__()

        self.features = nn.Sequential(OrderedDict([
                ('stemblock', _StemBlock(3, num_init_features)), 
            ]))     

        if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [bottleneck_width] * 4

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i])
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            num_trans_features = num_features
            if i != 0 and i != len(block_config) - 1:
                num_trans_features = int(num_features*compression_factor)
            elif i == len(block_config) - 1 and num_out_features is not None:
                num_trans_features = num_out_features

            self.features.add_module('transition%d' % (i + 1), BasicConv2d(num_features, num_trans_features, kernel_size=1))
            num_features = num_trans_features
            
            if i != len(block_config) - 1:
                self.features.add_module('transition%d_pool' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))

        self.use_se = use_se
        if self.use_se:
            self.seb = _SEBlock(num_features)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        if self.use_se:
            out = self.seb(out)

        return out



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':

    model = PeleeNet17b()
    print(model)

    def print_size(self, input, output):
        print(torch.typename(self).split('.')[-1], ' output size:',output.data.size())

    for layer in model.features:
        layer.register_forward_hook(print_size)

    # input_var = torch.autograd.Variable(torch.Tensor(1,3,540,960))
    # input_var = torch.autograd.Variable(torch.Tensor(1,3,1056,1920))
    # input_var = torch.autograd.Variable(torch.Tensor(1,3,960,5136))
    input_var = torch.autograd.Variable(torch.Tensor(1,3,180,180))
    output = model.forward(input_var)

