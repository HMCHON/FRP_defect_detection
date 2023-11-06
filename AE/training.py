import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def SetCallback(CheckpointPath, EarlyStoppingSet=False, ReduceLearningrateSet=False):
    CallbackList = []
    if EarlyStoppingSet == True: # Early Stopping
        early_stopping = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=10)
        CallbackList.append(early_stopping)

    if ReduceLearningrateSet == True:# Reduce Learning rate Plateau
        reduce = ReduceLROnPlateau(monitor='loss',
                                   factor=0.995,
                                   patience=5,
                                   verbose=1,
                                   mode='min')
        CallbackList.append(reduce)

    CheckpointFilePath = '%s%s%s' % (CheckpointPath, '/', 'Autoencoder_{epoch}.hdf5')
    checkpoint = ModelCheckpoint(filepath=CheckpointFilePath,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto')
    CallbackList.append(checkpoint)
    return CallbackList

def SaveHistory(Model, path):
    Loss = Model.history.history['loss']
    ValLoss = Model.history.history['val_loss']
    HistDF = pd.DataFrame({'loss':Loss,
                           'val_loss':ValLoss})

    HistDFName = '%s%s'%(path,'/history.csv')
    with open(HistDFName, mode='w') as f:
        HistDF.to_csv(f)
    return Loss, ValLoss

def PlotHistory(Loss, Val_loss, path):
    E = range(len(Loss))
    plt.plot(E, Loss, 'b')
    plt.plot(E, Val_loss, 'b')
    plt.title('Training and Test loss')
    fig2 = plt.gcf()
    fig2.savefig('%s%s' % (path, '/Train_and_Test_Loss'))
    plt.close(fig2)

def SaveEvalScore(model, ValDataset, SavePath, CNNmodelSelect):
    Score = model.evaluate(x=ValDataset, steps = 5)
    print("%s_%s: %.2f%%" % (CNNmodelSelect, model.metrics_names[1], Score[1] * 100))
    SaveResultPath = '%s%s' % (SavePath, '/Result.txt')
    with open(SaveResultPath, "w") as text_file:
        text_file.write("%s_%s: %.2f%%" % (CNNmodelSelect, model.metrics_names[1], Score[1] * 100))


def SaveConfusionMetric(model, ValDataset):
    ValDataset.reset()
    STEP_SIZE_VALID = ValDataset.n // ValDataset.batch_size
    Y_pred = model.predict(ValDataset, STEP_SIZE_VALID + 1)
    y_pred = np.argmax(Y_pred, axis=-1)
    ConfusionMetric = confusion_matrix(ValDataset.classes[ValDataset.index_array], y_pred)


    return ConfusionMetric

def SaveConfusionMetric2Img(ConfusionMetric, HistoryPath):
    labels = ['defect','non-defect']
    df_cm = pd.DataFrame(ConfusionMetric, range(2), range(2))
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams['figure.dpi'] = 300
    plt.pcolor(df_cm)
    plt.xticks(np.arange(0, 2, 1))
    plt.yticks(np.arange(0, 2, 1))
    sns.heatmap(df_cm,
                annot=True,
                fmt='d',
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={'size':23},
                cbar=False,
                cmap='Greens')
    plt.xlabel('Predicted Labels', fontsize=10)
    plt.ylabel('Actual Labels', fontsize=10)
    plt.title('Confusion Matrix', fontsize=10)
    plt.savefig(('%s%s'%(HistoryPath,'/ConfusionMatrix.jpg')))
