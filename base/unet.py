from prototype_classes import *
from data import *
from layers import *
from calcs import *


def get_DoubleConv(playground, in_channels, out_channels, mid_channels=None):
    DOUBLECONV = Integrated_Network_Block(playground)
    if not mid_channels:
        mid_channels = out_channels
    Conv2d_1 = Conv2d_Block(playground)
    Conv2d_1.set_parameters({'in_channels': in_channels, 'out_channels': mid_channels, 'kernel_size': (3, 3), 'stride': (1, 1),
                             'padding': (1, 1), 'dilation': (1, 1), 'groups': 1, 'bias': False, 'padding_mode': 'zeros',
                             'device': 'cuda', 'dtype': torch.float32})
    BatchNorm2d_1 = BatchNorm2d_Block(playground)
    BatchNorm2d_1.set_parameters({'num_features': mid_channels, 'eps': 1e-05, 'momentum': 0.1, 'affine': True,
                                  'track_running_stats': True, 'device': 'cuda', 'dtype': torch.float32})
    ReLU_1 = ReLU_Block(playground)
    ReLU_1.set_parameters({'inplace': True})

    Conv2d_2 = Conv2d_Block(playground)
    Conv2d_2.set_parameters(
        {'in_channels': mid_channels, 'out_channels': out_channels, 'kernel_size': (3, 3), 'stride': (1, 1),
         'padding': (1, 1), 'dilation': (1, 1), 'groups': 1, 'bias': False, 'padding_mode': 'zeros',
         'device': 'cuda', 'dtype': torch.float32})
    BatchNorm2d_2 = BatchNorm2d_Block(playground)
    BatchNorm2d_2.set_parameters({'num_features': out_channels, 'eps': 1e-05, 'momentum': 0.1, 'affine': True,
                                  'track_running_stats': True, 'device': 'cuda', 'dtype': torch.float32})
    ReLU_2 = ReLU_Block(playground)
    ReLU_2.set_parameters({'inplace': True})

    DOUBLECONV.add_layer(0, Conv2d_1)
    DOUBLECONV.add_layer(1, BatchNorm2d_1)
    DOUBLECONV.add_layer(2, ReLU_1)
    DOUBLECONV.add_layer(3, Conv2d_2)
    DOUBLECONV.add_layer(4, BatchNorm2d_2)
    DOUBLECONV.add_layer(5, ReLU_2)

    return DOUBLECONV

def get_Down(playground, in_channels, out_channels):
    DOWN = Integrated_Network_Block(playground)
    MaxPool2d = MaxPool2d_Block(playground)
    MaxPool2d.set_parameters({'kernel_size': 2, 'stride': None, 'padding': (0,), 'dilation': (1,), 'return_indices': False, 'ceil_mode': False})
    DoubleConv = get_DoubleConv(playground, in_channels, out_channels)
    DOWN.add_layer(0, MaxPool2d)
    DOWN.add_layer(1, DoubleConv)
    return DOWN

class Up(nn.Module):
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def get_Up(playground, in_channels, out_channels, bilinear=True):
    UP = Customized_Network_Block(playground)
    if bilinear:
        up = Upsample_Block(playground)
        up.set_parameters({'size': None, 'scale_factor': 2, 'mode': 'bilinear', 'align_corners': True, 'recompute_scale_factor': None})
        conv = get_DoubleConv(playground, in_channels, out_channels, in_channels // 2)
    else:
        up = ConvTranspose2d_Block(playground)
        up.set_parameters({'in_channels': in_channels, 'out_channels': in_channels // 2, 'kernel_size': (2, 2), 'stride': (2, 2), 'padding': (0, 0), 'output_padding': (0, 0), 'groups': 1, 'bias': True, 'dilation': (1, 1), 'padding_mode': 'zeros', 'device': 'cuda', 'dtype': torch.float32})
        conv = get_DoubleConv(playground, in_channels, out_channels)

    # idx_block = Idx_Block(playground)
    # tensor_size_block = Tensor_Size_Block(playground)

    code_block = Code_Block(playground)

    UP.set_order()



