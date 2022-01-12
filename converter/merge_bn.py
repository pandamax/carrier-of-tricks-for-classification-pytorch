import numpy as np
import torch

def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """ fuse convolution and batch norm's weight.
    Args:
        conv_w (torch.nn.Parameter): convolution weight.
        conv_b (torch.nn.Parameter): convolution bias.
        bn_rm (torch.nn.Parameter): batch norm running mean.
        bn_rv (torch.nn.Parameter): batch norm running variance.
        bn_eps (torch.nn.Parameter): batch norm epsilon.
        bn_w (torch.nn.Parameter): batch norm weight.
        bn_b (torch.nn.Parameter): batch norm weight.
    Returns:
        conv_w(torch.nn.Parameter): fused convolution weight.
        conv_b(torch.nn.Parameter): fused convllution bias.
    """
    import torch
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    # conv_w = conv_w * \
    #     (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    # conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    conv_w = torch.mul(conv_w, torch.mul(bn_w, bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1)))
    conv_b = (conv_b - bn_rm).mul(bn_var_rsqrt).mul(bn_w) + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def _fuse_conv_bn(conv, bn):
    conv.weight, conv.bias = \
        _fuse_conv_bn_weights(conv.weight, conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return conv


def _fuse_modules(model):
    r"""Fuses a list of modules into a single module
    Fuses only the following sequence of modules:
    conv, bn
    All other sequences are left unchanged.
    For these sequences, fuse modules on weight level, keep model structure unchanged.
    Arguments:
        model: Model containing the modules to be fused
    Returns:
        model with fused modules.
    """

    import torch

    children = list(model.named_children())
    conv_module = None
    conv_name = None

    for name, child in children:
       
        if isinstance(child, torch.nn.BatchNorm1d) or isinstance(
                child, torch.nn.BatchNorm2d) or isinstance(
                    child, torch.nn.BatchNorm3d):
            print('name',name, conv_module)
            print('child',child)
            conv_module = _fuse_conv_bn(conv_module, child)
            model._modules[conv_name] = conv_module
            child.eval()
            child.running_mean = child.running_mean.new_full(
                child.running_mean.shape, 0)
            child.running_var = child.running_var.new_full(
                child.running_var.shape, 1)
            if child.weight is not None:
                child.weight.data = child.weight.data.new_full(
                    child.weight.shape, 1)
            if child.bias is not None:
                child.bias.data = child.bias.data.new_full(child.bias.shape, 0)
            child.track_running_stats = False
            child.momentum = 0
            child.eps = 0
            conv_module = None
        elif isinstance(child, torch.nn.Conv2d) or isinstance(
                child, torch.nn.Conv3d):
            conv_module = child
            conv_name = name
        else:
            _fuse_modules(child)
    return model


def modify_prototxt_input(src_path):
    new_input_info = []
    model_info = []
    with open(src_path, 'r') as f:
        lines = f.readlines()

        model_info = lines[6:]
        input_h = lines[4].split(':')[-1]
        input_w = lines[5].split(':')[-1]

        new_input_info.append(lines[0])
        new_input_info.append('layer {\n')
        new_input_info.append('  name: "blob1"\n')
        new_input_info.append('  type: "Input"\n')
        new_input_info.append('  top: "blob1"\n')
        new_input_info.append('  input_param {\n')
        new_input_info.append('    shape {\n')
        new_input_info.append('      dim: 1\n')
        new_input_info.append('      dim: 3\n')
        new_input_info.append('      dim:' + input_h)
        new_input_info.append('      dim:' + input_w)
        new_input_info.append('    }\n')
        new_input_info.append('  }\n')
        new_input_info.append('}\n')

    with open(src_path, 'w') as f:
        for line in new_input_info:
            f.write(line)
        for line in model_info:
            f.write(line)