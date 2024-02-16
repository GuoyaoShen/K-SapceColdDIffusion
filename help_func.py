import numpy as np

import torch
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import time

def get_conv2d_dim(h_in: int, w_in: int, kernel_size: _size_2_t,
                   stride: _size_2_t = 1, padding: _size_2_t = 0,
                   dilation: _size_2_t = 1):
    """
    Args: function use to find the out dimension of a Conv2d layer

    :param h_in: input height, int
    :param w_in: input width, int
    :param kernel_size: kernel_size, int or tuple
    :param stride: stride, int or tuple, default = 1
    :param padding: padding, int or tuple, default = 0
    :param dilation: dilation, int or tuple, default = 1
    :return: tuple: out height dimension(H_out), out width dimension (W_out)
    """

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

    return int(h_out), int(w_out)


def get_conv_transpose2d_dim(h_in: int, w_in: int, kernel_size: _size_2_t,
                             stride: _size_2_t = 1, padding: _size_2_t = 0,
                             output_padding: _size_2_t = 0, dilation: _size_2_t = 1):
    """
    Args: function use to find the out dimension of a ConvTranspose2d layer

    :param h_in: input height, int
    :param w_in: input width, int
    :param kernel_size: kernel_size, int or tuple
    :param stride: stride, int or tuple, default = 1
    :param padding: padding, int or tuple, default = 0
    :param output_padding: padding, int or tuple, default = 0
    :param dilation: dilation, int or tuple, default = 1
    :return: tuple: out height dimension(H_out), out width dimension (W_out)
    """

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)

    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    w_out = (w_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

    return int(h_out), int(w_out)


def print_var_detail(var, name=""):
    """
    Args: Print basic detail of a variable

    :param var: input variable
    :param name: variable name, default is empty
    :return: string with basic information
    """
    print(name, "is a ", type(var), "with shape",
          var.shape if torch.is_tensor(var) or isinstance(var, np.ndarray) else None,
          "max: ",
          var.max() if torch.is_tensor(var) and not torch.is_complex(var) or isinstance(var, np.ndarray) else None,
          "min: ",
          var.min() if torch.is_tensor(var) and not torch.is_complex(var) or isinstance(var, np.ndarray) else None)


def num_to_groups(num, divisor):
    """
    Args:convert a num to a list of divisor, last element is its reminder

    :param num: input large number
    :param divisor: divisor is the num at the first len(num) - 1 element
    :return: a list of divisor, last element is its reminder
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))
