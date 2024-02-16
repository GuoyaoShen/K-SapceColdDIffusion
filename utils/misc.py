import numpy as np
import os

import torch



def calc_model_size(model):
    '''
    Calculate the model size.
    model: Pytorch models.
    '''
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def create_path(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    else:
        print("Path already exists.")
