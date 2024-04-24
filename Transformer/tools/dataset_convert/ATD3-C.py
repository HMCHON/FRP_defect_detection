import sys
import os
import glob
from Transformer.tools.utils.split import *
from Transformer.tools.utils.utils import *

import yaml
import math
import re

class create_dataset:
    def __init__(self, target_temp_path):
        with open('ATD3-C.yaml', 'r', encoding='utf-8') as f:
            self.conf = yaml.safe_load(f)  # 안전하게 YAML 로드
        self.height = self.conf['height']
        self.width = self.conf['width']
        self.patch_size = self.conf['patch_size']
        self.patch_stride = self.conf['patch_stride']

        if self.height == self.width and self.height == 256:
            matrices_4x4 = np.arange(self.height).reshape(self.patch_size, self.patch_size)
            self.matrices = print_and_collect_strided_matrices(matrices_4x4, self.patch_stride,
                                                               math.sqrt(self.patch_size))

        self.target_temp_path = os.listdir(target_temp_path)
        for target in sorted([file for file in self.target_temp_path if file.endswith('.csv')]):
            self.csv_file = self.target_temp_path  # 데이터셋으로 생성할 csv 파일을 여기에 넣기
            self.patches = split_csv_data_into_patches(self.height, f"{target_temp_path}/{target}", self.patch_size)
            temp_num = re.findall(r'\d+', target)[0]
            i = 0
            for matrix in self.matrices:
                area = transform_and_flatten(self.patches, matrix)
                create_directory(f'{target_temp_path}/T{temp_num}')
                save_to_npy(area, f'{target_temp_path}/T{temp_num}/T{i}A{i}.npy')
                save_to_npy(matrix, f'{target_temp_path}/T{temp_num}/T{i}M{i}.npy')
                i += 1



if __name__ == "__main__":

    base_path = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_2/'
    target_folder_name_list = ['Air_Defect_20_30_35_result',
                               'Air_Defect_50_result',
                               'None_Defect_result',
                               'Vacuum_Defect_20_30_35_result',
                               'Vacuum_Defect_40_result',
                               'Vacuum_Defect_50_result']
    """
     If you need to analyze ADT3 dataset, use the "process_all_txt_file_in_folder" function.
    """
    for target in target_folder_name_list:
        print(f"Analyzing {target} case ... ")
        node_csv_path = f"{base_path}/node_csv/{target}"
        process_all_txt_files_in_folder(target, node_csv_path)
        print(f"{target} case analysis completed")

    """
     Preprocessing ADT3 dataset (patching)
    """
    for target in target_folder_name_list:
        data_path = f"{base_path}/{target}"
        create_dataset(data_path)