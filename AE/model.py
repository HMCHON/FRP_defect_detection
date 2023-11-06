from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, ResNet50V2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense, BatchNormalization, Flatten


def AutoModel(model, image_size=225):

    if model == 'Autoencoder':
        # No concatenate
        input = layers.Input(shape=(image_size, image_size, 1))
        # Encoder
        ec_conv1 = layers.Conv2D(16, (15, 15), activation='relu', padding='same')(input)
        ec_max1 = layers.MaxPooling2D((3, 3), padding='same')(ec_conv1)
        ec_conv2 = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(ec_max1)
        ec_max2 = layers.MaxPooling2D((3, 3), padding='same')(ec_conv2)
        ec_conv3 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(ec_max2)
        ec_max3 = layers.MaxPooling2D((5, 5), padding='same')(ec_conv3)

        # Decoderplt.show()
        # plt.close()
        de_conv3 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(ec_max3)
        de_upda3 = layers.UpSampling2D((5, 5))(de_conv3)
        de_conv2 = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(de_upda3)
        de_upda2 = layers.UpSampling2D((3, 3))(de_conv2)
        de_conv1 = layers.Conv2D(16, (15, 15), activation='relu', padding='same')(de_upda2)
        de_upda1 = layers.UpSampling2D((3, 3))(de_conv1)
        output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(de_upda1)
        # Autoencoder
        autoencoder = Model(input, output)
        autoencoder.summary()
        BATCH_SIZE=16

    return autoencoder, BATCH_SIZE

def Compile(Model):
    model = Model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
