
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(120 * 3 * 3, 84)
        self.fc2 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)  # 드롭아웃 레이어 추가

    def forward(self, img):
        img = F.tanh(self.conv1(img))
        img = self.pool(img)
        img = F.tanh(self.conv2(img))
        img = self.pool(img)
        img = F.tanh(self.conv3(img))
        img = self.pool(img)
        img = img.view(img.size(0), -1)
        img = F.tanh(self.fc1(img))
        img = self.dropout(img)  # 드롭아웃 적용
        output = self.fc2(img)
        return output

class CustomMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super(CustomMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(p=0.5))  # 드롭아웃 레이어 추가
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output