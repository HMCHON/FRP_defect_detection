from dataset import *
import datetime
import os
from pathlib import Path
from tools import *
from dataset import *
from training import *
from PIL import Image
from model import *
import tensorflow as tf


def PREDICT(ImageList, model):
    MSEList = []
    AveMSE = 0
    for ImageDir in ImageList:
        IMG = Image.open(ImageDir)
        IMG = np.expand_dims(IMG, axis=0)
        Reconstruction = model.predict(IMG)
        MSE = tf.keras.losses.MeanSquaredError()
        MSEList.append(MSE(IMG, Reconstruction).numpy())
        AveMSE += MSE(IMG, Reconstruction).numpy()
    AveMSE = AveMSE/len(MSEList)
    return MSEList, AveMSE

def EvalMSE(CalMSE, VlMSEList):
    TrueCount=0
    for MSE in VlMSEList:
        if MSE < CalMSE:
            TrueCount = TrueCount+1
    return TrueCount



class EvaluateOnly:
    def __init__(self, AutomodelLoad, TrainDataDir, ValDataDir, imgSize):
        self.AutomodelLoad = AutomodelLoad
        self.TrainDataDir = TrainDataDir
        self.ValDataDir = ValDataDir
        self.imgSize = int(imgSize)
        self.HistoryPath = '%s%s'%((Path(self.AutomodelLoad).resolve().parent.parent), '/history')
        self.TrainDataset = LoadDatasetPath(self.TrainDataDir)
        self.ValDataset = LoadDatasetPath(self.ValDataDir)
        self.EvaluateModel()
    def EvaluateModel(self):
        self.LoadModel = LoadModel(self.AutomodelLoad)
        self.TrMSEList, self.TrAveMSE = PREDICT(self.ValDataset, self.LoadModel)
        self.VlMSEList, self.VlAveMSE = PREDICT(self.TrainDataset, self.LoadModel)
        self.TrueCount = EvalMSE(self.TrAveMSE, self.VlMSEList)
        print(self.TrueCount)




def set_variable():
    print('Chose CNN Model .hdf5')
    AutomodeLoad = PopFileChoose()
    print('Choose Train Directory')
    TrainDataDir = PopFileDir()
    print('Choose Validate Directory')
    ValDataDir = PopFileDir()
    print('Input Image Size')
    imgSize = input('Image Size:')

    EvaluateOnly(AutomodeLoad,
                 TrainDataDir,
                 ValDataDir,
                 imgSize)

def run():
    set_variable()

if __name__=="__main__":
    run()
