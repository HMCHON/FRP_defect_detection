import sys
import os
from Transformer.tools.utils import *

from easydict import EasyDict
import json
import math
from scipy.ndimage import zoom
import re
class create_dataset:
    def __init__(self, target_temp_path):
        self.conf = JsonConfigFileManager('FATD3-C.json')
        self.height = self.conf.values['height'] # 데이터 높이
        self.width = self.conf.values['width'] # 데이터 폭
        self.patch_size = self.conf.values['patch_size'] # 패치 크기
        self.patch_stride = self.conf.values['patch_stride'] # 패치 stride

        self.csv_file = "/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/tools/utils/resized_test.csv" # 데이터셋으로 생성할 csv 파일을 여기에 넣기

        # 패치 생성
        self.patches = split_csv_data_into_patches(self.height, self.csv_file, self.patch_size)
        # 에리어 생성 후 patch 위치 지정된 list 생성
        if self.height == self.width and self.height == 256:
            matrices_4x4 = np.arange(self.height).reshape(self.patch_size, self.patch_size)
            self.matrices = print_and_collect_strided_matrices(matrices_4x4, self.patch_stride, math.sqrt(self.patch_size))

        self.target_temp_path = os.listdir(target_temp_path)
        for target in [file for file in self.target_temp_path if file.endswith('.csv')]:
            i = 0
            temp_num = re.findall(r'\d+', target)[0]
            for matrix in self.matrices:
                area = transform_and_flatten(self.patches, matrix)
                create_directiry(f'{target_temp_path}/T{temp_num}')
                save_to_npy(area, f'{target_temp_path}/T{temp_num}/T{i}A{i}.npy')
                save_to_npy(matrix, f'{target_temp_path}/T{temp_num}/T{i}M{i}.npy')
                i += 1

if __name__ == "__main__":
    path = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_50_result'
    ds_creator = create_dataset(path)