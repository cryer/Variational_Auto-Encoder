import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import gzip

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

training_set=(
    read_image_file('data/raw/train-images-idx3-ubyte'),
    read_label_file('data/raw/train-labels-idx1-ubyte')
)
test_set=(
    read_image_file('data/raw/t10k-images-idx3-ubyte'),
    read_label_file('data/raw/t10k-labels-idx1-ubyte')
)
with open("data/processed/training.pt", 'wb') as f:
    torch.save(training_set, f)
with open("data/processed/test.pt", 'wb') as f:
    torch.save(test_set, f)

print('Done!')