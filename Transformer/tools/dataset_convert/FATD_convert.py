import sys
import os
import glob
from Transformer.tools.utils.split import *
from Transformer.tools.utils.FATD3_analyze import *
from Transformer.tools.utils.utils import *
from Transformer.tools.utils.normalized import *

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
        self.time_stamp = self.conf['time_stamp']
        self.base_path = target_temp_path

        if self.height == self.width and self.height == 256:
            matrices_4x4 = np.arange(self.height).reshape(self.patch_size, self.patch_size)
            self.matrices = print_and_collect_strided_matrices(matrices_4x4, self.patch_stride,
                                                               math.sqrt(self.patch_size))

        self.target_temp_path = os.listdir(target_temp_path)
        path_list = sorted([file for file in self.target_temp_path if file.endswith('extracted.csv')])

        temperature_data = pd.read_csv(f'{self.base_path}/temperature_data.csv')

        for idx, target in enumerate(path_list):
            temp_data = pd.read_csv(f"{target_temp_path}/{target}")
            data_array = temp_data.values

            # min, max값을 찾기 위한 칸
            start_col, end_col = idx+1, idx+self.time_stamp+1
            selected_columns = temperature_data.iloc[:, start_col:start_col + end_col]
            max_temp = selected_columns.max().max()
            min_temp = selected_columns.min().min()

            # 데이터 사이즈 맞추기 (64*64로)
            down_data1 = downsample_temperature_data_by_range(data_array,
                                                              grid_size=256,
                                                              method='min')
            down_data2 = downsample_temperature_data_by_range(data_array,
                                                              grid_size=256,
                                                              method='max')
            down_data3 = downsample_temperature_data_by_range(data_array,
                                                              grid_size=256,
                                                              method='mean')

            # Normalized data to min-max method and draw temperature picture
            down_data = min_max_normalize_3d_image(down_data1, down_data2, down_data3, min_temp, max_temp)
            save_to_png(down_data, target_temp_path, f"fig/{target}")
            #
            # temp_num = re.findall(r'\d+', target)[0]
            # i = 0
            # for matrix in self.matrices:
            #     area1 = transform_and_flatten(down_data, matrix)
            #     area2 = transform_and_flatten(down_data1, matrix)
            #     create_directory(f'{target_temp_path}/T{temp_num}')
            #     save_to_npy(area1, f'{target_temp_path}/T{temp_num}/T{i}A{i}.npy')
            #     save_to_npy(matrix, f'{target_temp_path}/T{temp_num}/T{i}M{i}.npy')
            #     save_to_npy(area2, f'{target_temp_path}/T{temp_num}/T{i}O{i}.npy')
            #     i += 1


if __name__ == "__main__":
    # 만약, 1HZ로 데이터를 전처리해야 한다면, 1hz_images.py 실행
    base_path = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3'
    target_folder_name_list = ['4-3']

    """
     If you need to analyze FADT3 dataset, use the "TemperatureAnalysis" function.
    """
    # for target in target_folder_name_list:
    #     print(f"Analyzing {target} case ... ")
    #     csv_path = f"{base_path}/{target}"
    #
    #     # # Ground truth를 기준으로 temperature data의 시편 부분 추출
    #     image_path = [file for file in os.listdir(csv_path) if file.endswith('.png')][0]
    #     extract_white_pixels(image_path, csv_path)
    #
    #     # 1차미분, 2차미분 구해서 graph로 plot.
    #     all_file_list = os.listdir(csv_path)
    #     file_list = sorted([file for file in all_file_list if file.endswith('extracted.csv')])
    #     for i in [0, 1, 2]:
    #         analysis = TemperatureAnalysis(target, csv_path, file_list, i, plot=False)
    #         analysis.run()
    #     print(f"{target} case analysis completed")

    """
     Preprocessing ADT3 dataset (patching)
    """
    for target in target_folder_name_list:
        data_path = f"{base_path}/{target}"
        create_dataset(data_path)