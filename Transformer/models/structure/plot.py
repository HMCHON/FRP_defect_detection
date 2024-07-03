import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 이미지 불러오기
img_path = 'stylized-image2.png'  # 여기에 이미지 파일 경로를 입력하세요
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 불러오기
data = np.array(img)

# X, Y 좌표 생성
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
x, y = np.meshgrid(x, y)

# 중심으로부터의 y축 방향 거리 계산
center_y = data.shape[0] // 2
dist_from_center_y = np.abs(y - center_y)

# y축을 중심으로부터 멀어질수록 거리의 제곱만큼 높이 증가
data_modified = data + (dist_from_center_y ** 2)

# 최대치를 원래 이미지의 최대치로 맞추기
original_max = data.max()
data_modified = data_modified * (original_max / data_modified.max())

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 플롯
ax.plot_surface(x, y, data_modified, cmap='gray')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Height')

plt.show()
