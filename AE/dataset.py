from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import tensorflow as tf

def KerasDataGenerate(path, imgSize, BatchSize):
    DataGenerate = ImageDataGenerator(validation_split=0.2)
    TrainDataFlowFromDir = DataGenerate.flow_from_directory(path,
                                                            shuffle=True,
                                                            target_size=(imgSize, imgSize),
                                                            batch_size=BatchSize,
                                                            class_mode='input',
                                                            subset='training')

    TestDataFlowFromDir = DataGenerate.flow_from_directory(path,
                                                           shuffle=True,
                                                           target_size=(imgSize, imgSize),
                                                           batch_size=BatchSize,
                                                           class_mode='input',
                                                           subset='validation')

    return TrainDataFlowFromDir, TestDataFlowFromDir

def ValDataGenerate(path, imgSize, BatchSize):
    DataGenerate = ImageDataGenerator()
    ValDataFlowFromDir = DataGenerate.flow_from_directory(path,
                                                          target_size=(imgSize, imgSize),
                                                          batch_size=BatchSize,
                                                          class_mode='input',
                                                          subset='training',
                                                          shuffle=False)
    return ValDataFlowFromDir

def LoadDatasetPath(path):
    file_list = os.listdir(path)
    AbsFileList = []
    for file in file_list:
        AbsPath = '%s%s%s'%(path,'/',file)
        AbsFileList.append(AbsPath)
    return AbsFileList

def TrainDataGenerate2npy(path):
    AbsNpyList = LoadDatasetPath(path)
    NpyArrayList = []
    for npy in AbsNpyList:
        NumpyArr = np.load(npy)
        NpyArrayList.append(NumpyArr)
    NpyArrayList = np.array(NpyArrayList)
    return NpyArrayList.astype(np.float32)


def DataTensorSlices(arr):
    dataset = tf.data.Dataset.from_tensor_slices(arr)
    return dataset