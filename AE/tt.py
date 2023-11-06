from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.models import Model
import PIL.Image as pilimg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error

'''
#0. Set GPU environment
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


reload_path = '/home/lams/PycharmProjects/GFRP_defect_detection/AE/Autoencoder_19.hdf5'
autoencoder = load_model(reload_path)
autoencoder.summary()
# No concatenate
encoder_input = autoencoder.layers[0]

# Encoder
ec_conv1 = autoencoder.layers[1]
ec_conv2 = autoencoder.layers[2]
ec_conv3 = autoencoder.layers[3]
ec_conv4 = autoencoder.layers[4]

# Decoder
de_conv3 = autoencoder.layers[5]
de_conv2 = autoencoder.layers[6]
de_conv1 = autoencoder.layers[7]
output = autoencoder.layers[8]

encoder = keras.Model(encoder_input.input, ec_conv4.output)
decoder = keras.Model(encoder_input.input, output.output)

image_path = '/home/lams/PycharmProjects/GFRP_defect_detection/AE/40F5_5_H_33.jpg'
image = Image.open(image_path)
image1 = np.array(image, dtype='float32')
image2 = image1/255.0
encoder_img = encoder.predict(image2[None])[0]
decoded_img = decoder.predict(image2[None])[0]
plt.imshow((decoded_img*255).astype(np.uint8))
plt.show()
plt.close()


plt.imshow((encoder_img*255).astype(np.uint8))
plt.show()
plt.close()

'''
path = '/home/lams/PycharmProjects/pythonProject/imageset/AE/val'
for j in os.listdir(path):
    file_folder = '%s%s%s' % (path,'/',j)
    for s in os.listdir(file_folder):
        file_dir = '%s%s%s' % (file_folder, '/', s)
        image = Image.open(file_dir)
        image = image.resize((1344,1344), Image.LANCZOS)
        image = np.array(image, dtype='float32')
        image /= 255.0
        encoded_imgs = encoder.predict(image[None])[0]
        decoded_imgs = decoder.predict(image[None])[0]

        plt.figure(figsize=(10,10))

        plt.subplot(1,4,1)
        plt.title('original')
        plt.imshow(image)

        plt.subplot(1,4,2)
        plt.title('encoder')
        plt.imshow((encoded_imgs*255).astype(np.uint8))

        plt.subplot(1,4,3)
        plt.title('decoder')
        plt.imshow((decoded_imgs*255).astype(np.uint8))

        plt.subplot(1,4,4)
        a = (np.square(image - decoded_imgs)).mean(axis=None)
        print(a)
        #plt.imshow((a*255).astype(np.uint8))


        plt.show()
        plt.close()
'''