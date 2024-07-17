import os
import re

def rename_files_in_directory(directory_path, num):
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            old_file_path = os.path.join(root, file_name)

            # 'Stats'가 포함된 파일 삭제
            if 'Stats' in file_name:
                os.remove(old_file_path)
                print(f'Removed: {old_file_path}')
                continue

            # 파일 이름에서 'aaa'를 'bbb'로 변경
            new_file_name = file_name.replace(f'{num}_', '')

            # 파일 이름에서 숫자를 추출하고 zfill로 4자리 숫자로 변경
            match = re.search(r'(\d+)', new_file_name)
            if match:
                number = match.group(1)
                new_number = number.zfill(3)
                new_file_name = new_file_name.replace(number, new_number)

            # 특정 문자열(num)을 파일 이름에서 제거
            if num in new_file_name:
                new_file_name = new_file_name.replace(f'{num}_', '')

            new_file_path = os.path.join(root, new_file_name)

            # 파일 이름이 변경된 경우에만 파일 이동(rename)
            if old_file_path != new_file_path:
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')

'''
,'1-2','1-3','1-4','1-5','1-6',
            '2-1','2-2','2-3','2-4','2-5','2-6',
            '3-1','3-2','3-3','3-4','3-5','3-6',
            '4-1','4-2','4-3','4-4','4-5','4-6',
            '5-1','5-2','5-3','5-4','5-5','5-6',
            '6-1','6-2','6-3','6-4','6-5','6-6',
            'N-1','N-2'
'''
idx_list = ['1-1']
for idx in idx_list:
    directory_path = f'/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_3/{idx}'
    rename_files_in_directory(directory_path, idx)

