import sys
import os
import glob

from Transformer.tools.utils.split import *
from Transformer.tools.utils.ATD3_analyze import *
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

        if self.height == self.width and self.height == 256:
            matrices_4x4 = np.arange(self.height).reshape(self.patch_size, self.patch_size)
            self.matrices = print_and_collect_strided_matrices(matrices_4x4, self.patch_stride,
                                                               math.sqrt(self.patch_size))

        self.target_temp_path = os.listdir(target_temp_path)
        path_list = sorted([file for file in self.target_temp_path if file.endswith('.csv')])

        # 현재 n초에서의 target csv에서부터 +a 초 까지의 temperature data 중 max, min을 찾아줄 temperature_data.csv load
        temperature_data = pd.read_csv(f'{target_temp_path}/temperature_data.csv')

        for idx, target in enumerate(path_list):
            temp_data = pd.read_csv(f'{target_temp_path}/{target}')
            data_array = temp_data.values
            x_coords, y_coords, temperatures = [row[1] for row in data_array], [row[2] for row in data_array], [row[3] for row in data_array]

            start_col, end_col = f'temperature_t{str(idx+1).zfill(3)}', f'temperature_t{str(idx+self.time_stamp+1).zfill(3)}'
            selected_columns = [col for col in temperature_data.columns if start_col <= col <= end_col]
            selected_data = temperature_data[selected_columns]
            max_temp = selected_data.max().max()
            min_temp = selected_data.min().min()

            # 데이터 사이즈 맞추기 (64*64로)
            down_data1 = downsample_temperature_data_by_physical_range(x_coords,
                                                                       y_coords,
                                                                       temperatures,
                                                                       physical_length=10.0,
                                                                       grid_size=64,
                                                                       method='min')
            down_data2 = downsample_temperature_data_by_physical_range(x_coords,
                                                                       y_coords,
                                                                       temperatures,
                                                                       physical_length=10.0,
                                                                       grid_size=64,
                                                                       method='max')
            down_data3 = downsample_temperature_data_by_physical_range(x_coords,
                                                                       y_coords,
                                                                       temperatures,
                                                                       physical_length=10.0,
                                                                       grid_size=64,
                                                                       method='mean')

            # Normalized data to min-max method and draw temperature picture
            down_data = min_max_normalize_3d_image(down_data1, down_data2, down_data3, min_temp, max_temp)
            save_to_png(down_data,
                        target_temp_path,
                        f"fig/{target}")
            temp_num = re.findall(r'\d+', target)[0]
            i = 0
            for matrix in self.matrices:
                area1 = transform_and_flatten(down_data, matrix)
                area2 = transform_and_flatten(down_data1, matrix)
                create_directory(f'{target_temp_path}/T{temp_num}')
                save_to_npy(area1, f'{target_temp_path}/T{temp_num}/T{i}A{i}.npy')
                save_to_npy(matrix, f'{target_temp_path}/T{temp_num}/T{i}M{i}.npy')
                save_to_npy(area2, f'{target_temp_path}/T{temp_num}/T{i}O{i}.npy')
                i += 1




if __name__ == "__main__":

    base_path = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_3'
    target_folder_name_list = ['Case_15']

    """
     If you need to analyze ADT3 dataset, use the "TemperatureAnalysis" function.
    """
    # for target in target_folder_name_list:
    #     print(f"Analyzing {target} case ... ")
    #     node_csv_path = f"{base_path}/node_csv/{target}.csv"
    #     txt_path = f"{base_path}/{target}"
    #     process_all_txt_files_in_folder(txt_path, node_csv_path) # node 위치에 알맞게 txt 파일 조정
    #
        # 1차미분, 2차미분 구해서 graph로 plot. 0=미분x, 1=1차미분, 2=2차미분, 3=그래프그리기x
        # data_path = f"{base_path}/{target}"
        # num_files = 360
        # for i in [0, 1, 2]:
        #     analysis = TemperatureAnalysis(data_path, data_path, num_files, i)
        #     analysis.run()
        # print(f"{target} case analysis completed")
        #
        # analysis = TemperatureAnalysis(data_path, data_path, num_files, 3)
        # analysis.run()


    """
     Preprocessing ADT3 dataset (patching)
    """
    for target in target_folder_name_list:
        data_path = f"{base_path}/{target}"
        create_dataset(data_path)