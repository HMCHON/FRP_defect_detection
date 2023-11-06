import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv
import os


def file_choose():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def load_csv(path):
    csv_file = open(path, 'r', encoding="utf_8", newline='')
    csv_file = csv.reader(csv_file)
    return csv_file


def extract_frames(csv_file, text='Frame'):
    text_range = len(text)
    frame = []
    check = 0
    for row in csv_file:
        try:
            if (row[0][0:text_range] == (text)) and check == 0:
                frame.append(row[1:])
                check = 1
            elif check == 1:
                frame.append(row[1:])
        except:
            pass
    return frame


def list2arr(list):
    arr = np.array(list)
    return arr


def merge_row(frame1, frame2):
    return np.append(frame1, frame2, axis=1)


def merge_column(frame1, frame2):
    frame3 = []
    frame3.append(frame1)
    frame3.append(frame2)
    return frame3


def merge_frame(frames, img_width=225, img_height=225, H=15):
    row = int(img_width / H)
    col = int(img_height / H)
    num = int(len(frames) / H)
    row_l = range(row, num, row)
    first = 0
    frame1 = list2arr(frames[0:row * H])
    for index in row_l:
        if index+H>num:
            pass
        else:
            frame2 = list2arr(frames[index * H:index * H + row * H])
            frame1 = merge_row(frame1, frame2)
    return frame1.T


def separate_rows(frames, num, increment, H=15):
    start = num
    end = num + increment * H
    return frames[start:end]


def separate_columns(frames, start_col, W=210, H=15):
    check = 0
    cframe = []
    for row in frames:
        start = start_col
        end = start_col + H
        if end > W:
            pass
        else:
            check += 1
            cframe.append(row[start:end])
    return cframe


def cal_defect(defect_x, start, H):
    if start <= defect_x and defect_x <= start + H:
        return True
    else:
        return False

def numpyArrayString2Float(arr):
    arr = arr.astype(np.float)
    return arr

def numpyArrayAddDim(arr):
    arr = np.expand_dims(arr, axis=2)
    return arr

def save2path(path, img):
    plt.imsave(path, img)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def CheckSize(arr, imgSize):
    if arr.shape == (imgSize,imgSize):
        return True
    else:
        return False

def def_direct(file_name, stride, count, fault=True):
    abs_path = file_name[:-17]
    if fault == True:
        npy_name = '%s%s%s%s%s%s' % (file_name[-8:-4], '_', stride, '_F_', count,'.jpg')
        path = '%s%s%s%s' % (abs_path, 'npy/Training/stride_',stride,'/defect')
        check_path(path)
        npy_save_pth = '%s%s%s' % (path, '/', npy_name)
    elif fault == False:
        npy_name = '%s%s%s%s%s%s' % (file_name[-8:-4], '_', stride, '_H_', count,'.jpg')
        path = '%s%s%s%s' % (abs_path, 'npy/Training/stride_',stride,'/non-defect')
        check_path(path)
        npy_save_pth = '%s%s%s' % (path, '/', npy_name)
    return npy_save_pth

def arr2img(arr):
    img = tf.keras.preprocessing.image.array_to_img(arr)
    return img

def separate_frames(file_name, frames, stride, defect_x, img_width=225, img_height=225, W=210, H=15):
    total_num = len(frames)
    num = int(img_width/H*img_height/H)
    num_list = range(0, int(total_num), num * H)
    st_list = range(0, W, stride)
    f_count, h_count = 0, 0
    for start1 in num_list:
        frame1 = separate_rows(frames, start1, num, H)
        for start2 in st_list:
            frame2 = separate_columns(frame1, start2, W, H)
            frame3 = merge_frame(frame2, img_width, img_height, H)
            Check = CheckSize(frame3, img_width)
            if Check == True:
                frame3 = numpyArrayAddDim(frame3)
                frame3 = numpyArrayString2Float(frame3)
                defect = cal_defect(defect_x, start2, H)
                if defect == True:
                    path = def_direct(file_name, stride, f_count, fault=defect)
                    f_count += 1
                elif defect == False:
                    path = def_direct(file_name, stride, h_count, fault=defect)
                    h_count += 1
                frame2img = arr2img(frame3)
                save2path(path, frame2img)
            else:
                pass

    return num_list
