import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# 이미지 로드 및 전처리
def load_image(img_path, transform=None, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')  # 이미지를 RGB 형식으로 변환
    if max_size:
        scale = max_size / max(image.size)
        size = int(scale * image.size[0]), int(scale * image.size[1])
        image = image.resize(size, Image.Resampling.LANCZOS)

    if shape:
        image = image.resize(shape, Image.Resampling.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image


# VGG 모델의 특정 레이어를 추출
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


# 그램 행렬 계산
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# 이미지 전처리 및 후처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 이미지 로드
content_path = '/mnt/data/a.png'
style_path = '/mnt/data/b.png'

content = load_image(content_path, transform)
style = load_image(style_path, transform, shape=[content.size(2), content.size(3)])

# VGG 모델 로드
vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content = content.to(device)
style = style.to(device)
vgg.to(device)

# 콘텐츠와 스타일 특징 추출
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# 스타일 특징의 그램 행렬 계산
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 결과 이미지 초기화 (콘텐츠 이미지로 초기화)
target = content.clone().requires_grad_(True).to(device)

# 옵티마이저 설정 (L-BFGS)
optimizer = optim.LBFGS([target])
criterion = nn.MSELoss()

# 스타일 전이 과정
style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
content_weight = 1e0  # alpha
style_weight = 1e5  # beta

run = [0]
while run[0] <= 1000:

    def closure():
        with torch.no_grad():
            target.clamp_(0, 1)

        optimizer.zero_grad()
        target_features = get_features(target, vgg)
        content_loss = criterion(target_features['conv4_2'], content_features['conv4_2'])

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * criterion(target_gram, style_gram)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f'Iteration {run[0]}, Total loss: {total_loss.item()}')

        return total_loss


    optimizer.step(closure)


# 후처리 함수
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


# 결과 이미지 출력
final_img = im_convert(target)
plt.imshow(final_img)
plt.axis('off')
plt.show()
