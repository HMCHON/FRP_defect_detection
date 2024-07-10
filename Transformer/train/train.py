from ..models.vanilla import ViT
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from load_data import *


# 동적으로 모델과 옵티마이저를 생성하는 함수
def create_dynamic_model_vars(n, ViT_mode, device):
    models = []
    optimizers = []
    epoch_losses = []
    logit_lists = []

    for i in range(n):
        if ViT_mode == 1:
            model = ViT1(ch=3,
                         img_size=64,
                         patch_size=16,
                         emb_dim=32,
                         n_layers=6,
                         out_dim=2,
                         dropout=0.1,
                         heads=2).to(device)
        elif ViT_mode == 2:
            model = ViT2(emb_dim=32,
                         time_step=9,
                         num_classes=2,
                         dropout=0.1,
                         n_layers=6,
                         heads=2).to(device)
        elif ViT_mode == 3:
            model = ViT3(emb_dim=32,
                         time_step=49,
                         num_classes=10,
                         dropout=0.1,
                         n_layers=3,
                         heads=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        model.train()

        models.append(model)
        optimizers.append(optimizer)
        epoch_losses.append([])
        logit_lists.append([])

    return models, optimizers, epoch_losses, logit_lists

def definee_eval_models(models_list):
    models = []

    for model in models_list:
        model.eval()
        models.append(model)

    return models

# 모델 훈련 및 평가 함수 예시
def train_and_evaluate_model0(model1_n, model2_n, epochs, time_step, base_path, save_path, endswithlist, emb_dim, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 동적으로 데이터셋 생성
    train_loaders, test_loaders = ([]), ([])
    for ends in endswithlist:
        train_loader, test_loader = load_all_datasets(base_path, ends, time_step, batch_size=batch_size)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    for i, loader in enumerate(train_loaders):
        print(f"Train Loader {i} has {len(loader)} batches")

    # 동적으로 모델과 옵티마이저 생성
    models1, optimizers1, epoch_losses1, logit_lists1 = create_dynamic_model_vars(model1_n, ViT_mode=1, device=device)
    models2, optimizers2, epoch_losses2, logit_lists2 = create_dynamic_model_vars(model2_n, ViT_mode=2, device=device)

    model3 = ViT3(emb_dim=emb_dim, time_step=model1_n, num_classes=2, dropout=0.1, n_layers=3, heads=2).to(device)
    model3.train()
    optimizer3 = optim.AdamW(model3.parameters(), lr=0.001)
    epoch_losses3 = []

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
    ###################### Train ViT1 models ######################
        for i in range(model1_n):
            model1 = models1[i]
            optimizer1 = optimizers1[i]

            for inputs, targets in train_loaders[i]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer1.zero_grad()
                logit1, outputs1 = model1(inputs)  # logit = cls_tkn
                loss = criterion(outputs1, targets)

                logit_lists1[i].append((logit1, targets))
                loss.backward()
                optimizer1.step()

                epoch_losses1[i].append(loss.item())

            if epoch % 5 == 0:
                print(f">>> ViT1_{i} : Epoch {epoch} train loss: ", np.mean(epoch_losses1[i]))
                epoch_losses1[i] = []

        # logit_lists1을 재구성하여 [49, num_batches * batch_size, emb_dim] 형태로 변환
        for i in range(model1_n):
            logits_and_targets = logit_lists1[i]
            logits = np.array([lt[0].detach().cpu().numpy() for lt in logits_and_targets]).reshape(-1, emb_dim)
            targets = np.array([lt[1].detach().cpu().numpy() for lt in logits_and_targets]).reshape(-1)
            logit_lists1[i] = (logits, targets)

    ###################### Train ViT2 models ######################
        for i in range(model2_n):
            model2 = models2[i]
            optimizer2 = optimizers2[i]

            logits, targets = logit_lists1[i]
            num_batches = len(logits) // time_step
            for batch_idx in range(num_batches):
                start_idx = batch_idx * time_step
                end_idx = start_idx + time_step
                inputs = torch.tensor(logits[start_idx:end_idx], device=device, requires_grad = False)  # [time_step, emb_dim]
                inputs = inputs.view(1, time_step, emb_dim)  # [1, time_step, emb_dim] - 1 is batch size in this case
                batch_targets = torch.tensor(targets[start_idx:end_idx], device=device, requires_grad = False)  # [time_step]

                optimizer2.zero_grad()
                logit2, outputs2 = model2(inputs)  # logit2.shape = [1, emb_dim], outputs2.shape = [1, num_classes]
                outputs2 = outputs2.view(time_step, -1)  # reshape to [time_step, num_classes]
                loss = criterion(outputs2, batch_targets.unsqueeze(0))

                majority_label = torch.mode(batch_targets).values.item()

                logit_lists2[i].append((logit2.detach().cpu().numpy(), majority_label))
                loss.backward()
                optimizer2.step()

                epoch_losses2[i].append(loss.item())

            if epoch % 5 == 0:
                print(f">>> ViT2_{i} : Epoch {epoch} train loss: ", np.mean(epoch_losses2[i]))
                epoch_losses2[i] = []

    ###################### Train ViT3 models ######################
        num_iterations = len(logit_lists2[0])
        for iteration in range(num_iterations):
            inputs = torch.tensor([logit_lists2[i][iteration][0] for i in range(model2_n)], device=device, requires_grad=False)  # [49, 32]
            inputs = inputs.view(1, model2_n, emb_dim)  # [1, 49, 32] - batch size is 1
            inputs_tensor = inputs.to(device)
            targets = torch.tensor([logit_lists2[i][iteration][1] for i in range(model2_n)], device=device, requires_grad=False)  # [49]

            optimizer3.zero_grad()
            logit3, outputs3 = model3(inputs_tensor)  # logit3.shape = [1, emb_dim], outputs3.shape = [1, num_classes]
            loss = criterion(outputs3, targets.mode().values.unsqueeze(0))  # loss with majority target

            loss.backward()
            optimizer3.step()

            epoch_losses3.append(loss.item())

            if epoch % 5 == 0:
                print(f">>> ViT3_{i} : Epoch {epoch} train loss: ", np.mean(epoch_losses3[i]))
                epoch_losses3[i] = []

        logit_lists1 = [[] for _ in range(model1_n)]
        logit_lists2 = [[] for _ in range(model2_n)]

        # 학습이 완료된 후 모델 가중치 저장
        for i, model in enumerate(models1):
            torch.save(model.state_dict(), f'{save_path}/ViT1_model_{i}.pth')
        for i, model in enumerate(models2):
            torch.save(model.state_dict(), f'{save_path}/ViT2_model_{i}.pth')
        torch.save(model3.state_dict(), f'{save_path}/ViT3_model.pth')

'''
def train_and_evaluate_model1(model1, models2_list, model3, train_dataloader, test_dataloader, device, epochs=1000):
    # Set model mode
    model1.train()
    models2 = define_eval_models(models2_list)
    model3.eval()

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            epoch_losses = []
            model.eval()
            for step, (inputs, labels) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                logit, outputs = model(inputs) # 여기서 logit과 outputs가 return됨
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))


def train_and_evaluate_model2(logit, model, train_dataloader, test_dataloader, device, epochs=1000):
    optimizer1 = optim.AdamW(model1.parameters(), lr=0.001)
    optimizer2 = optim.AdamW(model2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer1.zero_grad()
            logits, _ = model(inputs)  # ViT1의 로짓 x를 가져옵니다
            optimizer2.zero_grad()
            logit, outputs2 = model2(logits.unsqueeze(1))  # ViT2에 로짓 x를 입력으로 사용합니다
            loss = criterion(outputs2, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            epoch_losses = []
            model1.eval()
            model2.eval()
            for step, (inputs, labels) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model1(inputs)  # ViT1의 로짓 x를 가져옵니다
                _, outputs2 = model2(logits.unsqueeze(1))  # ViT2에 로짓 x를 입력으로 사용합니다
                loss = criterion(outputs2, labels)
                epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

def train_and_evaluate_model3(logit, model, train_dataloader, test_dataloader, device, epochs=1000):
    optimizer1 = optim.AdamW(model1.parameters(), lr=0.001)
    optimizer2 = optim.AdamW(model2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer1.zero_grad()
            logits, _ = model(inputs)  # ViT1의 로짓 x를 가져옵니다
            optimizer2.zero_grad()
            logit, outputs2 = model2(logits.unsqueeze(1))  # ViT2에 로짓 x를 입력으로 사용합니다
            loss = criterion(outputs2, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            epoch_losses = []
            model1.eval()
            model2.eval()
            for step, (inputs, labels) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model1(inputs)  # ViT1의 로짓 x를 가져옵니다
                _, outputs2 = model2(logits.unsqueeze(1))  # ViT2에 로짓 x를 입력으로 사용합니다
                loss = criterion(outputs2, labels)
                epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

'''

# 메인 함수
def main(mode, base_path, epochs, model1_n=49, model2_n=49, time_step=9): # 0=전체학습, 1=ViT1만 학습, 2=ViT2만 학습, 3=ViT3만 학습

    endsWithList = ['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg',
                    '12.jpg','13.jpg', '14.jpg','15.jpg','16.jpg','17.jpg','18.jpg','19.jpg','20.jpg','21.jpg','22.jpg',
                    '23.jpg','24.jpg','25.jpg', '26.jpg','27.jpg','28.jpg','29.jpg','30.jpg','31.jpg','32.jpg','33.jpg',
                    '34.jpg','35.jpg','36.jpg','37.jpg','38.jpg', '39.jpg','40.jpg','41.jpg','42.jpg','43.jpg','44.jpg',
                    '45.jpg','46.jpg','47.jpg', '48.jpg']

    if mode == 0: # 전체 ViT 학습 (ViT1 ~ ViT3)
        train_and_evaluate_model0(model1_n=model1_n,
                                  model2_n=model2_n,
                                  epochs=epochs,
                                  time_step=time_step,
                                  base_path=base_path,
                                  save_path='Transformer/models/weights',
                                  endsWithList=endsWithList,
                                  emb_dim=32,
                                  batch_size=32)

    # # 이미 학습된 .pth 파일이 있어야함
    # elif mode == 1: # ViT1만 학습
    #     model1 = ViT1().to(device)
    #     train_and_evaluate_model1()
    #
    # elif model == 2: # ViT2만 학습
    #     model2 = ViT2().to(device)
    #     train_and_evaluate_model2()
    #
    # elif mode == 3: # ViT3만 학습
    #     model3 = ViT3().to(device)
    #     train_and_evaluate_model3()


if __name__ == "__main__":
    mode = 0
    dataset_path = 'dataset_path'
    epochs = 10000

    # 데이터셋 경로 리스트
    base_path = "path/to/dataset1"

    main(mode, base_path, epochs)
