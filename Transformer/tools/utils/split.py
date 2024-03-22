import pandas as pd
import numpy as np

''' 256크기의 csv data를 patch로 나누는 함수 (256개의 patch 생성)'''
def split_256size_csv_data_into_patches(csv_file_path, patch_size):
    data = pd.read_csv(csv_file_path, header=None)
    data_array = data.values
    patch_size = patch_size
    patches = [data_array[i:i + patch_size, j:j + patch_size] for i in range(0, 256, patch_size) for j in
               range(0, 256, patch_size)]
    return patches # list

''' 전체 combinations의 갯수를 출력하는 함수 '''
def calculate_combinations(matrix_size, patch_size):
    start_points_per_side = matrix_size - patch_size + 1
    total_combinations = start_points_per_side ** 2
    return total_combinations


def get_strided_4x4_matrices(matrix, stride):
    patch_size = 4
    matrices = []
    for i in range(0, matrix.shape[0] - patch_size + 1, stride):
        for j in range(0, matrix.shape[1] - patch_size + 1, stride):
            patch = matrix[i:i + patch_size, j:j + patch_size]
            matrices.append(patch)

''' 스트라이드를 적용하여 4x4 크기의 행렬을 추출하고 출력하는 함수 '''
def print_and_collect_strided_4x4_matrices(matrix, stride):
    collected_matrices = get_strided_4x4_matrices(matrix, stride)
    for idx, matrix in enumerate(collected_matrices):
        print(f"Matrix #{idx + 1}:\n{matrix}\n")
    return collected_matrices

