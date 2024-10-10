#! /bin/python3

import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import torch_optimizer as optim
import lobs_utils

# 定义与之前相同的模型类
class SimpleDNN(lobs_utils.LobsDnnModel):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 28*28是输入图片的像素数，512是隐藏层的神经元数
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 10)    # 输出层，10个类别
        self.withReLUs = set(["fc1"])

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将图片展平成一维向量
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # 在激活函数后应用Dropout
        x = self.fc2(x)
        return x

def eval(model, test_dataset, test_loader):
  # 确保模型在评估模式
  model.eval()
  # 测试剪枝后的模型
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the pretrained network on the 10000 test images: {100 * correct / total} %')
    return correct / total

def train_epoch(model, train_loader):
    torch.set_grad_enabled(True)
    model.train()
    learning_rate = 0.01
    optimizer = optim.Adahessian(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        #loss.backward()
        loss.backward(create_graph=True)
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
# 检查CUDA是否可用
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# 加载保存的模型状态
model = SimpleDNN().to(device)
model.load_state_dict(torch.load('simple_dnn_mnist.pth'))
model.resetHessianStats()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
eval(model, test_dataset, test_loader)

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# 计算海赛矩阵
criterion = nn.CrossEntropyLoss()
print("Calculating hessian matrix...")
start_time = time.time()
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    inputs = images.view(-1, 28*28)
    lobs_utils.updateHessianStats(model, inputs)
    if (i % 50) == 0:
        print("Calc process:", model.sampleCount, "/", len(train_dataset))

model.eval()
h_pinvs = []
alpha = 10000.0
lobs_utils.calcHessiansAndPinvs(model, alpha)
end_time = time.time()
print("Generating hessian matrix and its pseudo-inverse done, ", end_time - start_time, " seconds elapsed.")

# 只剪第一层
layers = list(model.named_children())
for i, (name, layer) in enumerate(layers):
    if not isinstance(layer, nn.Linear):
        continue
    h = model.hessians[i]
    hinv = model.hpinvs[i]
    g_base = model.gradients[i]
    print("Layer: ", i, "name:", name, "Now begin to prune.")
    start_time = time.time()
    prune_seq_2d, loss_table_2d, accum_delta_w_table_2d = prePrune(model, layer, h, hinv, gbase)
    end_time = time.time()
    pre_prune_time = end_time - start_time
    start_time = time.time()
    original_weight, loss = greedyPrune(model, layer, 390000, prune_seq_2d, loss_table_2d, accum_delta_w_table_2d)
    end_time = time.time()
    prune_time = end_time - start_time
    acc = eval(model, test_dataset, test_loader)
    print("After ", 390000, " nodes surgeon, accurate=", acc, "layer loss=", loss, "pre prune time=", pre_prune_time, " seconds, prune_time=", prune_time, " seconds)
    break
