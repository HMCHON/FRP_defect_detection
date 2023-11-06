import datetime
import os
from tensorflow.keras.optimizers import Adam
from tools import *
from dataset import *
from training import *
from model import *

class trainingCNN:
    def __init__(self, GPU, Now, CNNmodelSelect, DataDir, imgSize, stride, epoch, EarlyStoppingSet, ReduceLearningrateSet):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU

        self.CNNmodelSelect = CNNmodelSelect
        self.DataDir = DataDir
        self.imgSize = int(imgSize)
        self.stride = int(stride)
        self.epoch = int(epoch)
        self.CheckpointPath = '%s%s%s%s' % (os.getcwd(), '/', Now, '/checkpoint')
        CheckPath(self.CheckpointPath)
        self.HistoryPath = '%s%s%s%s' % (os.getcwd(), '/', Now, '/history')
        CheckPath(self.HistoryPath)
        self.EarlyStoppingSet = bool(EarlyStoppingSet)
        self.ReduceLearningrateSet = bool(ReduceLearningrateSet)

        self.SelectCNNModel()
        self.GenerateDataset()
        self.TrainingModel()
        self.PlotTrainingModel()
        #self.EvaluateModel()

    def SelectCNNModel(self):
        self.SelectModel, self.BatchSize = AutoModel(self.CNNmodelSelect, self.imgSize)
        self.SelectModel.compile(optimizer='adam', loss='mse')


    def GenerateDataset(self):
        self.TrainDataset = TrainDataGenerate2npy(self.DataDir)
        self.TensorSliceTrain = DataTensorSlices(self.TrainDataset)


    def TrainingModel(self):
        CallbackList = SetCallback(self.CheckpointPath, self.EarlyStoppingSet, self.ReduceLearningrateSet)
        self.SelectModel.fit(x=self.TensorSliceTrain, y=None, batch_size=self.BatchSize)

    def PlotTrainingModel(self):
        Loss, ValLoss, = SaveHistory(self.SelectModel, self.HistoryPath)
        PlotHistory(Loss, ValLoss, self.HistoryPath)

    def EvaluateModel(self):
        SaveEvalScore(self.SelectModel, self.ValDataset, self.HistoryPath, self.CNNmodelSelect)
        ConfisionMetric = SaveConfusionMetric(self.SelectModel, self.ValDataset)
        SaveConfusionMetric2Img(ConfisionMetric, self.HistoryPath)


def set_variable():
    GPU = input('GPU:')
    Now = ('%s' % (datetime.datetime.now()))[0:-7]
    CNNmodelSelect = PopModelSelect()
    print('Choose Training Data Directory')
    DataDir = PopFileDir()
    imgSize = input('Image Size:')
    stride = input('Stride:')
    epoch = input('Epoch:')
    EarlyStoppingSet = input('Early Stopping Set (True or False):')
    ReduceLearningrateSet = input('Reduce Learning rate Set (True or False):')

    trainingCNN(GPU,
                Now,
                CNNmodelSelect,
                DataDir,
                imgSize,
                stride,
                epoch,
                EarlyStoppingSet,
                ReduceLearningrateSet)

def run():
    set_variable()

if __name__=="__main__":
    run()
