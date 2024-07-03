import pandas as pd
import numpy as np
from scipy import stats

''' 256크기의 csv data를 patch로 나누는 함수 (256개의 patch 생성)'''
def split_csv_data_into_patches(temp_data, patch_size):
    temp_size = temp_data.shape[0]
    patches = [temp_data[i:i + patch_size, j:j + patch_size] for i in range(0, temp_size, patch_size) for j in range(0, temp_size, patch_size)]
    return patches # list

''' 전체 combinations의 갯수를 출력하는 함수 '''
def calculate_combinations(matrix_size, patch_size):
    start_points_per_side = matrix_size - patch_size + 1
    total_combinations = start_points_per_side ** 2
    return total_combinations


def get_strided_4x4_matrices(matrix, stride, patch_size):
    matrices = []
    matrix = np.array(matrix)
    patch_size = int(patch_size)
    for i in range(0, matrix.shape[0] - patch_size + 1, stride):
        for j in range(0, matrix.shape[1] - patch_size + 1, stride):
            patch = matrix[i:i + patch_size, j:j + patch_size]
            matrices.append(patch)
    return matrices

''' 스트라이드를 적용하여 4x4 크기의 행렬을 추출하고 출력하는 함수 '''
def print_and_collect_strided_matrices(matrix, stride, patch_size):
    collected_matrices = get_strided_4x4_matrices(matrix, stride, patch_size)
    collect = []
    for idx, matrix in enumerate(collected_matrices):
        collect.append(matrix)
        # print(f"Matrix #{idx + 1}:\n{matrix}\n")
    return collect

''' 생성한 area 위치에 알맞게 patch를 넣어주는 함 '''
def transform_and_flatten(a, indices):
    if a.shape[0] == indices.shape[0]: # a데이터의 크기와 indices의 크기가 같은 경우
        temp_arr = a.flatten()
        restructured = []
        for row in indices:
            restructured_row = []
            for colloc in row:
                restructured_row.append(temp_arr[colloc])
            restructured.append(restructured_row)
        return restructured
    else:
        patches = split_csv_data_into_patches(a, 4)
        restructured_rows = []

        # Iterate over each row in indices
        for row in indices:
            restructured_row = [patches[colloc] for colloc in row]
            concatenated_row = np.concatenate(restructured_row, axis=1)
            restructured_rows.append(concatenated_row)

        # Concatenate all rows vertically
        final_concatenated = np.concatenate(restructured_rows, axis=0)
        return final_concatenated


''' temperature data가 들어있는 csv 파일을 64*64 크기(256개)로 변경 (ATD)'''
def downsample_temperature_data_by_physical_range(x_coords,
                                                  y_coords,
                                                  temperatures,
                                                  physical_length=10.0,
                                                  grid_size=64,
                                                  method='median'):
    """
    Downsample temperature data by physical range and calculate the specified statistic for each grid cell.

    Parameters:
    x_coords (np.ndarray): The x coordinates of the temperature data points.
    y_coords (np.ndarray): The y coordinates of the temperature data points.
    temperatures (np.ndarray): The temperature values at the given coordinates.
    physical_length (float): The physical length of one side of the area (default is 10.0 cm).
    grid_size (int): The number of cells along one edge of the grid (default is 16).
    method (str): The method to use for downsampling ('median', 'mean', 'max', 'min', 'mode').

    Returns:
    np.ndarray: The downsampled temperature data array with the specified statistic values.
    """
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    temperatures = np.array([float(temp.strip().replace('[', '').replace(']', '')) for temp in temperatures])


    # Define the physical size of each cell
    cell_size = physical_length / grid_size  # in cm

    # Initialize the result grid
    downsampled_data = np.zeros((grid_size, grid_size))

    # Iterate over each cell in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Define the bounds of the current cell
            x_min = -0.05 + i * cell_size * 10e-3
            x_max = -0.05 + (i + 1) * cell_size * 10e-3
            y_min = -0.05 + j * cell_size * 10e-3
            y_max = -0.05 + (j + 1) * cell_size * 10e-3

            # Find the indices of the points within the current cell
            indices = np.where(
                (x_coords >= x_min) & (x_coords < x_max) &
                (y_coords >= y_min) & (y_coords < y_max)
            )

            # Extract the temperatures within the current cell
            cell_temperatures = temperatures[indices]

            # Calculate the specified statistic for the points within the current cell
            if len(cell_temperatures) > 0:
                if method == 'median':
                    downsampled_data[i, j] = np.median(cell_temperatures)
                elif method == 'mean':
                    downsampled_data[i, j] = np.mean(cell_temperatures)
                elif method == 'max':
                    downsampled_data[i, j] = np.max(cell_temperatures)
                elif method == 'min':
                    downsampled_data[i, j] = np.min(cell_temperatures)
                elif method == 'mode':
                    mode_result = stats.mode(cell_temperatures)
                    downsampled_data[i, j] = mode_result.mode[0]
                else:
                    raise ValueError("Invalid method. Choose from 'median', 'mean', 'max', 'min', 'mode'.")
            else:
                downsampled_data[i, j] = np.nan  # Handle empty cells by assigning NaN

    return downsampled_data

''' temperature data가 들어있는 csv 파일을 64*64 크기(256개)로 변경 (FATD)'''
def downsample_temperature_data_by_range(temperatures, grid_size=64, method='median'):

    temperatures = np.array(temperatures)
    zoom_factor = temperatures.shape[0] / grid_size

    downsampled_data = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            # 현재 그리드의 영역 계산
            start_i, end_i = int(i * zoom_factor), int((i + 1) * zoom_factor)
            start_j, end_j = int(j * zoom_factor), int((j + 1) * zoom_factor)
            grid = temperatures[start_i:end_i, start_j:end_j]

            # 모드에 따라 해당 그리드의 값을 계산
            if method == 'median':
                downsampled_data[i, j] = np.median(grid)
            elif method == 'min':
                downsampled_data[i, j] = np.min(grid)
            elif method == 'max':
                downsampled_data[i, j] = np.max(grid)
            elif method == 'mean':
                downsampled_data[i, j] = np.mean(grid)

    return downsampled_data