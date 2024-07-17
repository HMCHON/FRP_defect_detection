import os
import shutil
#
# source_folder = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-5'
# destination_folder = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-5_1h'
#
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
#
# start_number = 155
# end_number = 9922
# frame_rate = 30
#
# for i in range(start_number, end_number + 1, frame_rate):
#     formatted_number = str(i).zfill(4)  # 숫자를 4자리로 포맷팅
#     source_file = os.path.join(source_folder, f'Temperature_{formatted_number}.csv')
#     destination_file = os.path.join(destination_folder, f'Temperature_{formatted_number}.csv')
#     if os.path.exists(source_file):
#         shutil.copy(source_file, destination_file)
#
# print('이미지 복사가 완료되었습니다.')

def copy_selected_csv_files(source_folder, destination_folder, start_number, end_number, frame_rate):
    """
    Copy specific CSV files from source_folder to destination_folder based on the given frame rate.

    Parameters:
    - source_folder (str): Path to the source directory containing the CSV files.
    - destination_folder (str): Path to the destination directory where selected files will be copied.
    - start_number (int): Starting number for the file selection.
    - end_number (int): Ending number for the file selection.
    - frame_rate (int): The interval at which files are selected and copied.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for i in range(int(start_number), int(end_number) + 1, frame_rate):
        formatted_number = str(i).zfill(4)  # Format the number to be 4 digits
        source_file = os.path.join(source_folder, f'{formatted_number}.csv')
        destination_file = os.path.join(destination_folder, f'{formatted_number}.csv')
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)

    print('File copying completed.')

'''
# Example usage
source_folder = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-5'
destination_folder = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-5_1h'
start_number = 155
end_number = 9922
frame_rate = 30

copy_selected_csv_files(source_folder, destination_folder, start_number, end_number, frame_rate)
'''
