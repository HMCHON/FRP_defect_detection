import csv
import pandas as pd
import numpy as np
import os
from scipy.ndimage import zoom

def save_to_csv(data, file_name):
    """
    데이터를 받아 CSV 파일로 저장하는 함수.
    Parameters:
    - data: 리스트의 리스트 형태의 데이터. 첫 번째 리스트는 헤더(열 이름)를 포함한다.
    - file_name: 저장될 CSV 파일의 전체 경로와 이름.
    """
    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv(file_name, index=False)
    print(f'{file_name}에 데이터가 저장되었습니다.')

def save_to_npy(data, file_name):
    """
    데이터를 받아 (np.array) npy 파일로 저장하는 함수
    :param data: npy 파일로 만들고자 하는 data(np.array)
    :param file_name: 저장할 파일 이름
    """
    np.save(file_name, data)
    print(f'{file_name}에 데이터가 저장되었습니다.')

def create_directiry(directory):
    """
    경로를 받아 있는지 확인하고, 경로가 없으면 생성하는 함수
    :param directory: 생성할 경로
    :return: 없
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def re_index_temperature_data_based_on_csv(음temperature_data_path, csv_data_name_with_path):
    """
    온도 데이터와 csv 파일의 위치 데이터를 매칭하는 함수.
    :param temperature_data_path: 온도 데이터가 저장되어 있는 txt 파일의 경로 ['node', 'temperature']
    :param csv_data_name_with_path: 노드 위치가 저장되어 있는 csv 데이터의 경로
    """
    raw_temp_data = pd.read_csv(temperature_data_path, header=None)
    raw_temp_data = raw_temp_data.values.flatten()
    nodes = raw_temp_data[1::2].astype(int)
    temperatures = raw_temp_data[2::2]
    temperature_data = pd.DataFrame({'node': nodes, 'temperature': temperatures})

    csv_data = pd.read_csv(csv_data_name_with_path, header=None, names=['x', 'y', 'z', 'node'])
    csv_data['node'] = pd.to_numeric(csv_data['node'], errors='coerce')
    sorted_csv_data = csv_data.sort_values(by=['x', 'y'])

    sorted_temperature_data = temperature_data.set_index('node').reindex(index=sorted_csv_data['node']).reset_index()

    sorted_csv_data.to_csv(csv_data_name_with_path.replace('.csv', '_sorted.csv'), index=False)
    sorted_temperature_data.to_csv(temperature_data_path.replace('.txt', '_sorted.csv'), index=False,
                                   sep=',', columns=['node', 'temperature'])


def process_all_txt_files_in_folder(folder_path, csv_data_name_with_path):
    """
    주어진 폴더 내의 모든 txt 파일에 대해 온도 데이터 재인덱싱 함수를 적용하는 함수.
    :param folder_path: 검색할 폴더 경로
    :param csv_data_name_with_path: 노드 위치가 저장되어 있는 csv 데이터의 경로
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            temperature_data_path = os.path.join(folder_path, file_name)
            re_index_temperature_data_based_on_csv(temperature_data_path, csv_data_name_with_path)


# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_20_30_35_result/Defect_Air_20x20_30x30_35x35.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_20_30_35_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('Air_Defect_20_30_35')
#
# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_50_result/Defect_Air_50x50.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_50_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('Air_Defect_50')
#
# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_40_result/Defect_Air_40x40.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_40_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('Air_Defect_40')
#
# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/None_Defect_result/None_Defect.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/None_Defect_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('None_Defect')
#
# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_20_30_35_result/Defect_Empty_20x20_30x30_35x35.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_20_30_35_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('Vacuum_Defect_20_30_35')
#
# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_40_result/Defect_Empty_40x40.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_40_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('Vacuum_Defect_40')
#
# csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_50_result/Defect_Empty_50x50.csv'
# temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_50_result'
# process_all_txt_files_in_folder(temp_name, csv_name)
# print('Vacuum_Defect_50')


def create_square_matrix_each_temperature_data_ansys_version(temperature_data_csv, row1, row2):
    df = pd.read_csv(temperature_data_csv)
    chunks = []
    current_chunk = []
    chunk_size = row1
    row_counter = 0
    for index, row in df.iterrows():
        current_chunk.append(float(row.iloc[1].replace(']', '').strip()))
        row_counter += 1
        if row_counter == chunk_size:
            if chunk_size == row2:
                modified_chunk = []
                for i in range(len(current_chunk) - 1):
                    modified_chunk.append(current_chunk[i])
                    average = 0.5 * (current_chunk[i] + current_chunk[i + 1])
                    modified_chunk.append(average)
                modified_chunk.append(current_chunk[-1])
                current_chunk = modified_chunk
            chunks.append(np.array(current_chunk))
            current_chunk = []
            row_counter = 0
            chunk_size = row2 if chunk_size == row1 else row1
    if current_chunk:
        chunks.append(np.array(current_chunk))
    return chunks


def chunks_to_2d_array(chunks, array_shape=(221, 221)):
    # 전체 데이터를 저장할 빈 배열을 생성
    full_array = np.zeros(array_shape)
    # 현재 위치를 추적
    current_row = 0
    current_col = 0
    for chunk in chunks:
        for value in chunk.flatten():  # 청크를 1D 배열로 변환하여 순회
            full_array[current_row, current_col] = value
            current_col += 1
            if current_col == array_shape[1]:  # 현재 행이 가득 차면 다음 행으로 이동
                current_row += 1
                current_col = 0
    return full_array


def save_2d_array_to_csv(data_array, output_csv):
    df = pd.DataFrame(data_array)
    df.to_csv(output_csv, index=False, header=False)


def resize_csv_data(csv_file_path, target_shape=(256, 256)):
    """
    지정된 CSV 파일에서 데이터를 불러오고, 해당 데이터를 지정된 크기로 조정합니다.

    Parameters:
    - csv_file_path: str. 데이터를 불러올 CSV 파일의 경로입니다.
    - target_shape: tuple. 조정될 데이터의 타겟 크기입니다. 기본값은 (255, 255)입니다.

    Returns:
    - resize_temp_data: numpy.ndarray. 조정된 데이터가 담긴 NumPy 배열입니다.
    """
    # CSV 파일에서 데이터 불러오기
    data = np.genfromtxt(csv_file_path, delimiter=',')
    # 데이터가 비어있지 않은지 확인
    if data.size == 0:
        raise ValueError("데이터가 비어있습니다.")
    # 스케일 계산
    scale_x = target_shape[0] / data.shape[0]
    scale_y = target_shape[1] / data.shape[1]
    # 데이터 조정
    resize_temp_data = zoom(data, (scale_x, scale_y))
    return resize_temp_data



# tmp = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Temperature 1_sorted.csv'
# row1 = 221
# row2 = 111
# chunk = create_square_matrix_each_temperature_data_ansys_version(tmp, row1, row2)
# chunks = chunks_to_2d_array(chunk)
# save_2d_array_to_csv(chunks, 'test.csv')
# save_2d_array_to_csv(resize_csv_data('/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/tools/utils/test.csv'), 'resized_test.csv')



