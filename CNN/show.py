import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

a = np.load('/home/lams/PycharmProjects/GFRP_defect_detection/dataset/npy/Training/stride_5/defect/40F1_5_F_0.npy')
b = np.load('/home/lams/PycharmProjects/GFRP_defect_detection/dataset/npy/Training/stride_3/defect/40F1_3_F_16.npy')

from PIL import Image
img_data = np.random.random(size=(100, 100, 1))
img = tf.keras.preprocessing.image.array_to_img(b)
array = tf.keras.preprocessing.image.img_to_array(img)

img = plt.imsave('test.jpg' , img)
