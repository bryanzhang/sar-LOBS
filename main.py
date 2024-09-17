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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
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
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
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
lobs_utils.calcHessiansAndPinvs(model)
end_time = time.time()
print("Generating hessian matrix and its pseudo-inverse done, ", end_time - start_time, " seconds elapsed.")

# 逐层剪枝
STEP=390000
TOPK=390000
layers = list(model.named_children())
for i, (name, layer) in enumerate(layers):
    if not isinstance(layer, nn.Linear):
        continue
    h = model.hessians[i]
    print("Layer: ", i, "name:", name, "Now begin to prune.")
    with torch.no_grad():
        for j in range(0, TOPK, STEP):
            count = j + STEP
            original_weight = layer.weight.data.clone()
            #print("Sample weight before prune:", original_weight[(375129 // layer.in_features)][(375129 % layer.in_features)])
            indices = lobs_utils.prune_fcn_layer(layer, count)
            assert(indices.size(0) == count)

            w = original_weight.reshape(layer.in_features * layer.out_features).clone()
            #print("Sample weight before prune 1:", original_weight[(375129 // layer.in_features)][(375129 % layer.in_features)])
            prune_mask = torch.zeros(layer.in_features * layer.out_features, dtype=torch.bool)
            prune_mask.index_fill_(0, indices, True)
            w.masked_fill_(prune_mask, 0.0)
            layer.weight.data = w.reshape(layer.out_features, layer.in_features)
            acc = eval(model, test_dataset, test_loader)
            print("After simply pruning by weight magnitude, acc=", acc)
            #print("Sample weight after prune:", original_weight[(375129 // layer.in_features)][(375129 % layer.in_features)])

            start_time = time.time()
            train_epoch(model, train_loader)
            end_time = time.time()
            torch.set_grad_enabled(False)
            #print("Sample weight after prune 1:", original_weight[(375129 // layer.in_features)][(375129 % layer.in_features)])
            acc = eval(model, test_dataset, test_loader)
            print("Retrain after pruning", count, "weights, accurate=", acc, ",execution time:", (end_time - start_time), "seconds.")
            #print("Sample weight after prune 2:", original_weight[(375129 // layer.in_features)][(375129 % layer.in_features)])
            layer.weight.data = original_weight.clone()

            start_time = time.time()
            assert(indices.size(0) == count)
            hpinv = model.hpinvs[i]
            gama = 100000
            weight, loss, original_delta = lobs_utils.optimal_brain_surgeon_v2(layer, indices, h, gama)
            #print("Sample weight after prune 3:", original_weight[(375129 // layer.in_features)][(375129 % layer.in_features)])
            end_time = time.time()
            layer.weight.data = weight
            acc = eval(model, test_dataset, test_loader)
            flat_weight = original_weight.flatten()
            original_obs_weights = flat_weight + original_delta.squeeze(1)
            selected_values = torch.masked_select(original_obs_weights, prune_mask)
            abs_vals = torch.abs(selected_values)
            avg_abs = torch.mean(abs_vals, dim=0)
            min_abs = torch.min(abs_vals, dim=0)
            max_abs = torch.max(abs_vals, dim=0)
            #print("Sample weight after LOBS:", layer.weight.data[(375129 // layer.in_features)][(375129 % layer.in_features)])
            print("LOBS pruned ", count, "weights, after surgeon, accurate=", acc, "layer loss=", loss, ",exection time:", (end_time - start_time), "seconds," + "Avg abs:", avg_abs, ", min abs:", min_abs, "max_abs:", max_abs, "L2 norm of original weight:", torch.norm(flat_weight, p=2), "L2 norm of original delta:", torch.norm(original_delta, p=2), "After surgeon, L2 norm is:", torch.norm(layer.weight.data.flatten(), p=2))
            layer.weight.data = original_weight
    break
