from dataset import *
import datetime
import os
from pathlib import Path
from tools import *
from dataset import *
from training import *
from model import *

class EvaluateOnly:
    def __init__(self, Now, CNNmodelSelect, CNNmodelLoad, ValDataDir, imgSize):
        self.CNNmodelSelect = CNNmodelSelect
        self.CNNmodelLoad = CNNmodelLoad
        self.ValDataDir = ValDataDir
        self.imgSize = int(imgSize)
        self.HistoryPath = '%s%s'%((Path(self.CNNmodelLoad).resolve().parent.parent),'/history')
        self.SelectModel, self.BatchSize = CNNModel(self.CNNmodelSelect, 2, image_size=self.imgSize)
        self.ValDataset = ValDataGenerate(self.ValDataDir,self.imgSize,self.BatchSize)
        self.EvaluateModel()
    def EvaluateModel(self):
        self.LoadModel = LoadModel(self.CNNmodelLoad)
        SaveEvalScore(self.LoadModel, self.ValDataset, self.HistoryPath, self.CNNmodelSelect)
        ConfisionMetric = SaveConfusionMetric(self.SelectModel, self.ValDataset)
        SaveConfusionMetric2Img(ConfisionMetric, self.HistoryPath)


def set_variable():
    Now = ('%s' % (datetime.datetime.now()))[0:-7]
    print('Select CNN Model')
    CNNmodelSelect = PopModelSelect()
    print('Chose CNN Model .hdf5')
    CNNmodelLoad = PopFileChoose()
    print('Choose Validate Directory')
    ValDataDir = PopFileDir()
    print('Input Image Size')
    imgSize = input('Image Size:')

    EvaluateOnly(Now,
                 CNNmodelSelect,
                 CNNmodelLoad,
                 ValDataDir,
                 imgSize)

def run():
    set_variable()

if __name__=="__main__":
    run()
