import numpy as np
import matplotlib.pyplot as plt

def robust_normalized_3d_image(data1, data2, data3):
    combined_data = np.stack((data1, data2, data3), axis=-1)

    # 각 채널을 robust normalization
    normalized_data_robust = np.zeros_like(combined_data)

    for i in range(combined_data.shape[-1]):
        median_val = np.median(combined_data[:, :, i])
        iqr_val = np.percentile(combined_data[:, :, i], 75) - np.percentile(combined_data[:, :, i], 25)
        normalized_data_robust[:, :, i] = (combined_data[:, :, i] - median_val) / iqr_val

    # 0~255 사이로 스케일링
    min_r = np.min(normalized_data_robust)
    max_r = np.max(normalized_data_robust)
    normalized_data_robust = (normalized_data_robust - min_r) / (max_r - min_r) * 255
    normalized_data_robust = np.round(normalized_data_robust).astype(np.uint8)

    return normalized_data_robust

def min_max_normalize_3d_image(data1, data2, data3, min_temp, max_temp):
    combined_data = np.stack((data1, data2, data3), axis=-1)

    # 각 채널별로 Min-Max 정규화
    normalized_data_minmax = np.zeros_like(combined_data)

    for i in range(combined_data.shape[-1]):
        normalized_data_minmax[:, :, i] = (combined_data[:, :, i] - min_temp) / (max_temp - min_temp)

    # 0~255 사이로 스케일링
    normalized_data_minmax = normalized_data_minmax * 255
    normalized_data_minmax = np.clip(normalized_data_minmax, 0, 255)  # 값의 범위를 0~255로 제한
    normalized_data_minmax = np.round(normalized_data_minmax).astype(np.uint8)

    return normalized_data_minmax


def save_to_png(array, path, name):
    # 3차원 이미지를 RGB로 시각화 (여백, 축, 그리드 없음)
    fig, ax = plt.subplots(figsize=(array.shape[1], array.shape[0]), dpi=80)
    ax.imshow(array)
    ax.axis('off')  # x축, y축 제거
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 여백 제거
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # x축 눈금 제거
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # y축 눈금 제거
    plt.gca().xaxis.set_ticks_position('none')  # x축 눈금 위치 제거
    plt.gca().yaxis.set_ticks_position('none')  # y축 눈금 위치 제거
    plt.gca().set_frame_on(False)  # 프레임 제거

    # 이미지 저장
    save_path = f'{path}/{name}.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
