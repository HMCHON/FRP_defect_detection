from tensorflow.keras.preprocessing.image import ImageDataGenerator


def KerasDataGenerate(path, imgSize, BatchSize):
    DataGenerate = ImageDataGenerator(validation_split=0.2)
    TrainDataFlowFromDir = DataGenerate.flow_from_directory(path,
                                                            shuffle=True,
                                                            target_size=(imgSize, imgSize),
                                                            batch_size=BatchSize,
                                                            class_mode='binary',
                                                            subset='training')
    TestDataFlowFromDir = DataGenerate.flow_from_directory(path,
                                                           shuffle=True,
                                                           target_size=(imgSize, imgSize),
                                                           batch_size=BatchSize,
                                                           class_mode='binary',
                                                           subset='validation')
    return TrainDataFlowFromDir, TestDataFlowFromDir

def ValDataGenerate(path, imgSize, BatchSize):
    DataGenerate = ImageDataGenerator()
    ValDataFlowFromDir = DataGenerate.flow_from_directory(path,
                                                          target_size=(imgSize, imgSize),
                                                          batch_size=BatchSize,
                                                          class_mode='binary',
                                                          subset='training',
                                                          shuffle=False)
    return ValDataFlowFromDir