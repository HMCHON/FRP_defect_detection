import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub


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

def tensor_to_image(tensor):
    tensor = tensor * 225
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 256
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def reduce_deviation(array, factor=0.5):
    median = np.median(array)  # 배열의 중앙값을 계산
    adjusted_array = median + factor * (array - median)  # 편차 줄이기
    return adjusted_array
def add_noise_3d_image(data, style_name, weight1=0.5, weight2=0.5):
    # Stage 1: Adapt scale factor matrix based on distance from the center
    center_x = data.shape[0] // 2
    center_y = data.shape[1] // 2

    scale_factors = np.zeros((data.shape[0], data.shape[1]))

    mean_data = np.mean(data)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if data[x,y] > mean_data:
                distance_y = (y-center_y)
                scale_factors[x,y]  = (1/(center_y)**2) * (distance_y ** 2)
    modified_data = data * scale_factors

    # Stage 2: Using a style transfer model to generate noise
    style_image = load_img(f'content_imgs/{style_name}') # style_image를 어떻게 설정할 지 좀 고려해보기
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(style_image), tf.constant(style_image))[0]
    stylized_pil_image = tensor_to_image(stylized_image)

    noise_image = stylized_pil_image.convert('L')
    min_val = np.min(noise_image)
    max_val = np.max(noise_image)
    noise_array = np.array(noise_image)
    original_noise_array = np.clip(noise_array, min_val, max_val)
    adjusted_noise_array = reduce_deviation(original_noise_array, factor=0.5)  # factor 값을 더 크게 설정

    # Stage 3: adjusted_noise_array + modified_data and add noise based on gaussian distribution
    weighted_array = (weight1 * modified_data + weight2 * adjusted_noise_array)

    random_factors = np.random.uniform(1, 1.2, (data.shape[0], data.shape[1]))
    weighted_array = weighted_array * random_factors
    weighted_array = np.clip(weighted_array, data.min(), data.max())

    return weighted_array

def save_to_png(array, path, name):
    # 3차원 이미지를 RGB로 시각화 (여백, 축, 그리드 없음)
    fig, ax = plt.subplots(figsize=(array.shape[1], array.shape[0]), dpi=1) # 1:1 scale로 저장
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
