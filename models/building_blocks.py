import torch
import numpy as np 


def make_activation(function_type):
    if function_type == 'relu': 
        return torch.nn.ReLU(inplace=True)
    elif function_type == 'leaky_relu':
        return torch.nn.LeakyReLU(inplace=True)
    elif function_type == 'elu':
        return torch.nn.ELU(inplace=True)
    elif function_type == 'gelu':
        return torch.nn.GELU()
    else:
        raise NotImplementedError(f'activation type {function_type} undefined in code')


def init_layers(layer):
    # https://github.com/SaoYan/DnCNN-PyTorch/blob/master/utils.py
    
    layer_name = layer.__class__.__name__ 
    if layer_name.find('Conv') != -1: 
        torch.nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
    elif layer_name.find('BatchNorm') != -1: 
        # what are these numbers and where they came from?????????
        layer.weight.data.normal_(mean=0, std=np.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        torch.nn.init.constant_(layer.bias.data, 0.0)
    return None


class ChannelAttention(torch.nn.Module): 
    
    # https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf
    pass 


class ConvLayer(torch.nn.Sequential): 
    pass


class DenseBlock(torch.nn.Sequential): 
    pass


class Down(torch.nn.Sequential):
    
    def __init__(self, channels):
        super(Down, self).__init__()
        self.add_module('Down Sample Max Pool', torch.nn.MaxPool2d(kernel_size=2))


class Up(torch.nn.Module):
    
    def __init__(self, channels, attention=False, interpolation=None, transpose=None, **kwargs):
        super(Up, self).__init__()
        assert any([interpolation, transpose]) and not all([interpolation, transpose])
        if isinstance(interpolation, str):
            assert transpose is None 
            self.up = torch.nn.Upsample(scale_factor=2, mode=interpolation, align_corners=True)
        elif isinstance(transpose, bool):
            assert interpolation is None
            self.up = torch.nn.ConvTranspose2d(in_channels=channels, out_channels=channels, 
                                               kernel_size=2, stride=2)

    def forward(self, lower_level, skip_connection):
        lower_level = self.up(lower_level)
        
        diff_x = skip_connection.shape[-1] - lower_level.shape[-1]
        diff_y = skip_connection.shape[-2] - lower_level.shape[-2]
        # pad the inputs
        lower_level = torch.nn.functional.pad(lower_level, [diff_x // 2, diff_x - diff_x // 2,
                                                            diff_y // 2, diff_y - diff_y // 2])
        
        lower_level = torch.cat([lower_level, skip_connection], dim=1)
        return lower_level

