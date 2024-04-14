import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from template.dataset import MNIST
from template.model import LeNet5, CustomMLP
from torchvision import transforms
import os
import pandas as pd
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, sample in enumerate(trn_loader):
        images, labels = sample['image'].to(device), sample['label'].to(device)  # 'labels' -> 'label'로 변경
        labels = labels.long()  # labels를 long 텐서로 변환
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    trn_loss /= len(trn_loader)
    acc = 100 * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()
    tst_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tst_loader:
            images, labels = sample['image'].to(device), sample['label'].to(device)  # 'labels' -> 'label'로 변경
            labels = labels.long()  # labels를 long 텐서로 변환
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            tst_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    tst_loss /= len(tst_loader)
    acc = 100 * correct / total

    return tst_loss, acc

def extract_label(file_name):
    return int(file_name.split('_')[1].split('.')[0])  # 파일 이름에서 레이블 추출

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환하고 픽셀 값 범위를 [0, 1]로 조정
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # 주어진 평균과 표준편차로 정규화
    ])

#   DataFrame 생성
    img_dir_train = "./data/train"
    file_names_train = os.listdir(img_dir_train)    

    img_dir_test = "./data/test"
    file_names_test = os.listdir(img_dir_test)  

    df_train = pd.DataFrame({'file_name': file_names_train})
    df_test = pd.DataFrame({'file_name': file_names_test})

    # 'label' 컬럼 추가
    df_train['label'] = df_train['file_name'].apply(extract_label)
    df_test['label'] = df_test['file_name'].apply(extract_label)

    # 'file_name'을 index로 설정
    df_train.set_index('file_name', inplace=True)
    df_test.set_index('file_name', inplace=True)

    # Instantiate dataset and data loaders
    
    trn_dataset = MNIST(df=df_train, img_dir="./data/train", transform=transform)
    tst_dataset = MNIST(df=df_test, img_dir="./data/test", transform=transform)

    trn_loader = DataLoader(trn_dataset, batch_size=256, shuffle=True)
    tst_loader = DataLoader(tst_dataset, batch_size=256, shuffle=False)

    # Instantiate models
    lenet5 = LeNet5().to(device)
    custom_mlp = CustomMLP(input_size=784, hidden_sizes=[128, 64], num_classes=10).to(device)

    # Define optimizer and cost function
    optimizer_lenet5 = optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    optimizer_mlp = optim.SGD(custom_mlp.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    lenet5_trn_losses = []
    lenet5_trn_accs = []
    lenet5_tst_losses = []
    lenet5_tst_accs = []
    mlp_trn_losses = []
    mlp_trn_accs = []
    mlp_tst_losses = []
    mlp_tst_accs = []

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        print("Training LeNet-5...")
        trn_loss_lenet5, trn_acc_lenet5 = train(lenet5, trn_loader, device, criterion, optimizer_lenet5)
        tst_loss_lenet5, tst_acc_lenet5 = test(lenet5, tst_loader, device, criterion)
        print(f"LeNet-5 - Train Loss: {trn_loss_lenet5:.4f}, Train Acc: {trn_acc_lenet5:.2f}%, Test Loss: {tst_loss_lenet5:.4f}, Test Acc: {tst_acc_lenet5:.2f}%")
        lenet5_trn_losses.append(trn_loss_lenet5)
        lenet5_trn_accs.append(trn_acc_lenet5)
        lenet5_tst_losses.append(tst_loss_lenet5)
        lenet5_tst_accs.append(tst_acc_lenet5)
        
        print("Training CustomMLP...")
        trn_loss_mlp, trn_acc_mlp = train(custom_mlp, trn_loader, device, criterion, optimizer_mlp)
        tst_loss_mlp, tst_acc_mlp = test(custom_mlp, tst_loader, device, criterion)
        print(f"CustomMLP - Train Loss: {trn_loss_mlp:.4f}, Train Acc: {trn_acc_mlp:.2f}%, Test Loss: {tst_loss_mlp:.4f}, Test Acc: {tst_acc_mlp:.2f}%")
        mlp_trn_losses.append(trn_loss_mlp)
        mlp_trn_accs.append(trn_acc_mlp)
        mlp_tst_losses.append(tst_loss_mlp)
        mlp_tst_accs.append(tst_acc_mlp)
        
        print()

    # 학습 결과 시각화
    plt.figure(figsize=(12, 4))
    epochs = range(1, num_epochs + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, lenet5_trn_losses, label='LeNet-5 Train Loss')
    plt.plot(epochs, lenet5_tst_losses, label='LeNet-5 Test Loss')
    plt.plot(epochs, mlp_trn_losses, label='CustomMLP Train Loss')
    plt.plot(epochs, mlp_tst_losses, label='CustomMLP Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, lenet5_trn_accs, label='LeNet-5 Train Acc')
    plt.plot(epochs, lenet5_tst_accs, label='LeNet-5 Test Acc')
    plt.plot(epochs, mlp_trn_accs, label='CustomMLP Train Acc')
    plt.plot(epochs, mlp_tst_accs, label='CustomMLP Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()