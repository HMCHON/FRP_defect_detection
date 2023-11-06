from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, ResNet50V2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense, BatchNormalization, Flatten


def CNNModel(model, nb, image_size=225, Weight=None):

    if model == 'VGG-19':
        base_model = VGG19(include_top = False,
                           weights = Weight,
                           input_shape = (image_size,image_size,3),
                           classes=nb,
                           classifier_activation='softmax')
        BATCH_SIZE = 64
    if model == 'Resnet152V2':
        base_model = ResNet152V2(include_top = False,
                                 weights = Weight,
                                 input_shape = (image_size,image_size,3),
                                 classes=nb,
                                 classifier_activation='softmax')
        BATCH_SIZE = 32
    if model == 'Densenet201':
        base_model = DenseNet201(include_top = False,
                                 weights = Weight,
                                 input_shape = (image_size,image_size,3),
                                 classes=nb)
        BATCH_SIZE = 16
    if model == 'Resnet50':
        base_model = ResNet50V2(include_top = False,
                                weights = Weight,
                                input_shape = (image_size,image_size,3),
                                classes=nb,
                                classifier_activation='softmax')
        BATCH_SIZE = 32

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    predictions = Dense(1, activation='softmax')(x)
    model = Model(base_model.input, outputs=predictions)
    model.summary()
    return model, BATCH_SIZE

def Compile(Model):
    model = Model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
