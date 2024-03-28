from ..utils import *

from easydict import EasyDict
import json
import math

class create_dataset:
    def __init__(self):
        self.conf = JsonConfigFileManager('FATD3-C.json')
        self.height = self.conf.values['height'] # 데이터 높이
        self.width = self.conf.values['width'] # 데이터 폭
        self.patch_size = self.conf.values['patch_size'] # 패치 크기
        self.patch_stride = self.conf.values['patch_stride'] # 패치 stride

        self.csv_file = "" # 데이터셋으로 생성할 csv 파일을 여기에 넣기
        # 패치 생성
        self.patches = split_csv_data_into_patches(self.height, self.csv_file, self.patch_size)
        # 에리어 생성 후 patch 위치 지정된 list 생성
        if self.height == self.width and self.height == 256:
            matrices_4x4 = np.arange(self.height).reshape(self.patch_size, self.patch_size)
            self.matrices = print_and_collect_strided_matrices(matrices_4x4[:3], self.patch_stride, math.sqrt(self.patch_size))
        # 패치 위치에 알맞게 패치 배치 후 csv로 output
        new_csv = []
        for idx, matrix in self.matrices:
            area = transform_and_flatten(self.patches, matrix)
            new_csv.append(area)
        # 생성된 csv(new_csv)를 f"T{second}F{frame_num(1~30)}"으로 저장
        save_to_csv(new_csv, 'name_here')

if __name__ == "__main__":
    create_dataset()
