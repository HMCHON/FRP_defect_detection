import sys
import os
import glob
import yaml
import math
import re
import gc
import numpy as np
import pandas as pd

# Add the base directory to the system path
sys.path.append('/media/lams/D/PycharmProjects/FRP_defect_detection')  # This should be the base directory containing the Transformer module

from Transformer.tools.utils.split import *
from Transformer.tools.utils.ATD3_analyze import *
from Transformer.tools.utils.utils import *
from Transformer.tools.utils.normalized import *


class create_dataset:
    def __init__(self, target_temp_path, name):
        with open('/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/tools/dataset_convert/ATD3-C.yaml', 'r', encoding='utf-8') as f:
            self.conf = yaml.safe_load(f)  # 안전하게 YAML 로드
        self.height = self.conf['height']
        self.width = self.conf['width']
        self.patch_size = self.conf['patch_size']
        self.patch_stride = self.conf['patch_stride']
        self.time_stamp = self.conf['time_stamp']

        if self.height == self.width and self.height == 256:
            matrices_4x4 = np.arange(self.height * self.width).reshape(self.height, self.width)
            self.matrices = print_and_collect_strided_matrices(matrices_4x4, self.patch_stride,
                                                               self.patch_size)

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

            # 데이터 사이즈 맞추기 (256*256으로)
            down_data1 = downsample_temperature_data_by_physical_range(x_coords,
                                                                       y_coords,
                                                                       temperatures,
                                                                       physical_length=10.0,
                                                                       grid_size=256,
                                                                       method='min')
            down_data2 = downsample_temperature_data_by_physical_range(x_coords,
                                                                       y_coords,
                                                                       temperatures,
                                                                       physical_length=10.0,
                                                                       grid_size=256,
                                                                       method='max')
            down_data3 = downsample_temperature_data_by_physical_range(x_coords,
                                                                       y_coords,
                                                                       temperatures,
                                                                       physical_length=10.0,
                                                                       grid_size=256,
                                                                       method='mean')

            # Normalized data to min-max method and draw temperature picture
            down_data_ori = min_max_normalize_3d_image(down_data1, down_data2, down_data3, min_temp, max_temp)
            directory_path = f"{target_temp_path}/fig"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            save_to_png(down_data_ori,
                        target_temp_path,
                        f"fig/O{target}")

            # Add noise to 3ch down_data
            down_data = add_noise_3d_image(down_data_ori, name, weight1=0.7, weight2=0.3)
            save_to_png(down_data,
                        target_temp_path,
                        f'fig/C{target}')

            temp_num = re.findall(r'\d+', target)[0]
            i = 0
            for matrix in self.matrices:
                directory_path = f"{target_temp_path}/T{temp_num}"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                area1 = transform_and_flatten(down_data, matrix)
                # area2 = transform_and_flatten(down_data1, matrix)
                # create_directory(f'{target_temp_path}/T{temp_num}')
                # save_to_npy(area1, f'{target_temp_path}/T{temp_num}/TA{i}.npy')
                # save_to_npy(matrix, f'{target_temp_path}/T{temp_num}/TM{i}.npy')
                # save_to_npy(area2, f'{target_temp_path}/T{temp_num}/TO{i}.npy')
                save_to_png(area1, target_temp_path, f'T{temp_num}/TI{i}')
                i += 1

            # 메모리 해제 및 가비지 컬렉션
            del temp_data, data_array, x_coords, y_coords, temperatures, down_data1, down_data2, down_data3, down_data_ori, down_data, area1
            gc.collect()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_ATD_dataset.py <base_path> <target>")
        sys.exit(1)

    base_path = sys.argv[1]
    target = sys.argv[2]
    data_path = os.path.join(base_path, target)
    print(data_path)
    create_dataset(data_path, target)

    # 메모리 해제 및 가비지 컬렉션
    del data_path
    gc.collect()
