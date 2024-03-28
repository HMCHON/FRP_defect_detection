import pandas as pd
import numpy as np

''' 256크기의 csv data를 patch로 나누는 함수 (256개의 patch 생성)'''
def split_csv_data_into_patches(img_size, csv_file_path, patch_size):
    data = pd.read_csv(csv_file_path, header=None)
    data_array = data.values
    patch_size = patch_size
    patches = [data_array[i:i + patch_size, j:j + patch_size] for i in range(0, img_size, patch_size) for j in
               range(0, img_size, patch_size)]
    return patches # list

''' 전체 combinations의 갯수를 출력하는 함수 '''
def calculate_combinations(matrix_size, patch_size):
    start_points_per_side = matrix_size - patch_size + 1
    total_combinations = start_points_per_side ** 2
    return total_combinations


def get_strided_4x4_matrices(matrix, stride, patch_size):
    matrices = []
    for i in range(0, matrix.shape[0] - patch_size + 1, stride):
        for j in range(0, matrix.shape[1] - patch_size + 1, stride):
            patch = matrix[i:i + patch_size, j:j + patch_size]
            matrices.append(patch)

''' 스트라이드를 적용하여 4x4 크기의 행렬을 추출하고 출력하는 함수 '''
def print_and_collect_strided_matrices(matrix, stride, patch_size):
    collected_matrices = get_strided_4x4_matrices(matrix, stride, patch_size)
    for idx, matrix in enumerate(collected_matrices):
        print(f"Matrix #{idx + 1}:\n{matrix}\n")
    return collected_matrices

''' 생성한 area 위치에 알맞게 patch를 넣어주는 함 '''
def transform_and_flatten(a, indices):
    # 재구성된 배열을 초기화합니다.
    restructured = []

    # 입력 배열 a의 차원에 따라 적절한 처리를 수행합니다.
    for row in indices:
        if isinstance(a[0], list):  # a의 원소가 리스트일 경우 (2차원 이상)
            restructured_row = []
            for index in row:
                restructured_row.extend(a[index])
            restructured.append(restructured_row)
        else:  # a의 원소가 정수일 경우 (1차원)
            restructured.append([a[index] for index in row])

    # 최종 결과를 반환합니다.
    return restructured
