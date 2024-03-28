import csv
import pandas as pd
import numpy as np
import os

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


def re_index_temperature_data_based_on_csv(temperature_data_path, csv_data_name_with_path):
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


csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_20_30_35_result/Defect_Air_20x20_30x30_35x35.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_20_30_35_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('Air_Defect_20_30_35')

csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_50_result/Defect_Air_50x50.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_50_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('Air_Defect_50')

csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_40_result/Defect_Air_40x40.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Air_Defect_40_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('Air_Defect_40')

csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/None_Defect_result/None_Defect.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/None_Defect_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('None_Defect')

csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_20_30_35_result/Defect_Empty_20x20_30x30_35x35.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_20_30_35_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('Vacuum_Defect_20_30_35')

csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_40_result/Defect_Empty_40x40.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_40_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('Vacuum_Defect_40')

csv_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_50_result/Defect_Empty_50x50.csv'
temp_name = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/row_data/Vacuum_Defect_50_result'
process_all_txt_files_in_folder(temp_name, csv_name)
print('Vacuum_Defect_50')




