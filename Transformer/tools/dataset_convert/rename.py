import os
import re

path = '/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/FATD3/4-2'
files = os.listdir(path)

for file in files:
    if 'Stats' in file:
        os.remove(os.path.join(path, file))
    if not 'Temperature' in file:
        if file.endswith('.csv'):
            parts = file.split('_')
            name = parts[1]
            num = name.split('.')

            # num = file.split('.')
            num = num[0]
            new_num = num.zfill(4)

            src = os.path.join(path, file)
            dst = os.path.join(path, f"Temperature_{new_num}.csv")
            os.rename(src, dst)
