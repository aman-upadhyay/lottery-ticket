import time

import numpy as np
#import tensorflow as tf
from tqdm import tqdm


def weight():
    w = np.load("mnist_conv2D/mnist_conv2D_data/trial3/10/same_init/final/layer3.npy")
    print(w)
    print(w.shape)


def no_of_weights_left(x):
    y = 100
    print(0, x)
    for h in range(1, 30):
        y = y - 0.2 * y
        print(h, (y * x) / 100)


# no_of_weights_left(300)

pbar = tqdm(total=100, ascii=".>=", desc="Pruning (1 iteration is 1 epoch)")
for x in range(100):
    pbar.update()
    time.sleep(0.1)
pbar.close()