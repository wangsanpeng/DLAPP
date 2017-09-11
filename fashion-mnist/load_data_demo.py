from utils import mnist_reader
from matplotlib import pyplot as plt
import numpy as np

import time

# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

label_map = {0:'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
height, width = 28, 28

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
num = X_train.shape[0]

for index in range(num):
    image = X_train[index,:].reshape((height, width))
    label = label_map[y_train[index]]
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.show()
    plt.clf()
