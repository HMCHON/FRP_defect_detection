import os
import shutil

source_folder = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-5'
destination_folder = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-5_1h'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

start_number = 155
end_number = 9922
frame_rate = 30

for i in range(start_number, end_number + 1, frame_rate):
    formatted_number = str(i).zfill(4)  # 숫자를 4자리로 포맷팅
    source_file = os.path.join(source_folder, f'Temperature_{formatted_number}.csv')
    destination_file = os.path.join(destination_folder, f'Temperature_{formatted_number}.csv')
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)

print('이미지 복사가 완료되었습니다.')
