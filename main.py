# Machine learning: Mnist Project
# image recognition
# Mnist --> recognizing handwritten digits

import numpy as np
import keras
from keras.datasets import mnist
from matplotlib import pyplot

# loading the training and testing set into separate variables
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# p
print(train_x.shape) # x is the image
print(train_y.shape) # y is the lable (the answer of the image, the num)
print(test_x.shape)
print(test_y.shape)

# display sample dataset
# for i in range(9):
#     pyplot.subplot(i)
#     pyplot.imshow(train_x[i], cmap = pyplot.get_cmap('grey'))
#     pyplot.show()

