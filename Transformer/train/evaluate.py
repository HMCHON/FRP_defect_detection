# 학습이 완료된 후
model.eval()  # 평가 모드로 설정
with torch.no_grad():
    inputs, _ = next(iter(test_dataloader))
    inputs = inputs.to(device)
    features = model(inputs, return_features=True)
    print(f"Feature map size: {features.size()}")
