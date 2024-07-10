import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def data_loader(file_paths, labels, time_step, batch_size):
    """
    데이터 로더 생성 함수
    :param file_paths: 이미지 파일 경로의 리스트
    :param labels: 각 이미지에 대한 레이블 리스트
    :return: DataLoader 객체
    배치 크기를 고려하여 데이터셋 나누기:
    """
    transform = transforms.Compose([ # 필요한 경우 이미지 크기 조정
        transforms.ToTensor()
    ])
    dataset = ImageDataset(file_paths, labels, transform=transform)

    # dataset split
    total_size = len(dataset)
    train_size = (total_size // batch_size) * batch_size
    test_size = total_size - train_size

    indices = torch.randperm(total_size).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=time_step, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=time_step, shuffle=False)

    return train_loader, test_loader

def find_specific_files(base_path, time_step, pattern="O48.npy"):
    """
    base_path 디렉토리 아래에서 특정 패턴으로 끝나는 파일들을 찾는 함수
    각 상위 폴더에서 찾은 파일의 개수가 time_step으로 나누어 떨어지도록 필터링합니다.
    :param base_path: 탐색을 시작할 디렉토리 경로
    :param time_step: 필터링 기준이 되는 time_step 값
    :param pattern: 찾고자 하는 파일의 패턴 (기본값은 "O48.npy")
    :return: 해당 패턴으로 끝나는 파일들의 경로 리스트
    """
    matching_files = []
    folder_files = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(pattern):
                folder = os.path.basename(root)
                if folder not in folder_files:
                    folder_files[folder] = []
                folder_files[folder].append(os.path.join(root, file))

    for files in folder_files.values():
        files = sorted(files)
        num_files = len(files)
        remainder = num_files % time_step
        valid_count = num_files - remainder
        if valid_count > 0:
            matching_files.extend(files[:valid_count])
            for f in files[:valid_count]:
                folder_name = os.path.basename(os.path.dirname(f))
                if folder_name in ['case1', 'case2', 'N-1', 'N-2']:
                    label = "Non-defect"
                else:
                    label = "Defect"
                labels.append(label)
    return matching_files, labels

def load_all_datasets(base_path, pattern, time_step, batch_size):
    """
    데이터셋을 로드하는 모든 함수를 실행시키는 함수
    :param base_path: 데이터셋이 위치한 기본 경로
    :param pattern: 찾고자 하는 파일의 패턴
    :return: DataLoader 객체
    """
    matching_file_list, labels = find_specific_files(base_path, time_step, pattern)
    train_loader, test_loader = data_loader(matching_file_list, labels, time_step, batch_size=batch_size)
    return train_loader, test_loader

