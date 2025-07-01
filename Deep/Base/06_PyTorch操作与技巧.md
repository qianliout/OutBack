# 06_PyTorch操作与技巧

## PyTorch张量操作详解

# PyTorch张量操作详解

## 1. 张量拼接与堆叠

### 1.1 torch.cat() - 张量拼接

`torch.cat()` 用于**沿指定维度拼接多个张量**，不会新增维度，而是直接在现有维度上扩展数据。

#### 函数定义
```python
torch.cat(tensors, dim=0, *, out=None) → Tensor
```

#### 核心规则
1. **所有张量的形状必须相同**（除了拼接维度 `dim` 的大小可以不同）
2. **拼接维度外的其他维度必须完全一致**
3. **不会新增维度**，而是在现有维度上扩展

#### 使用示例

```python
import torch

# 沿第0维拼接（垂直堆叠）
x = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
y = torch.tensor([[5, 6]])          # shape: (1, 2)
z = torch.cat([x, y], dim=0)        # 沿行方向拼接
print(z)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

# 沿第1维拼接（水平堆叠）
x = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
y = torch.tensor([[5], [6]])        # shape: (2, 1)
z = torch.cat([x, y], dim=1)        # 沿列方向拼接
print(z)
# tensor([[1, 2, 5],
#         [3, 4, 6]])

# 高维张量拼接
x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
y = torch.randn(2, 1, 4)  # shape: (2, 1, 4)
z = torch.cat([x, y], dim=1)  # 沿第1维拼接
print(z.shape)  # torch.Size([2, 4, 4])
```

### 1.2 torch.stack() - 张量堆叠

`torch.stack()` 用于**将多个张量沿新维度堆叠**，会创建一个新的维度。

#### 函数定义
```python
torch.stack(tensors, dim=0, *, out=None) → Tensor
```

#### 核心规则
1. **所有张量的形状必须完全相同**
2. **会新增一个维度**，新维度的大小等于输入张量的数量
3. **堆叠后的张量维度数 = 输入张量维度数 + 1**

#### 使用示例

```python
import torch

# 堆叠两个向量（新增第0维）
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.stack([x, y])  # dim=0
print(z)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(z.shape)  # torch.Size([2, 3])

# 指定 dim=1 新增维度
z = torch.stack([x, y], dim=1)
print(z)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
print(z.shape)  # torch.Size([3, 2])

# 堆叠矩阵（新增第3维）
x = torch.randn(2, 3)  # shape: (2, 3)
y = torch.randn(2, 3)
z = torch.stack([x, y], dim=2)  # 新增第3维
print(z.shape)  # torch.Size([2, 3, 2])
```

### 1.3 cat vs stack 对比

| 方法 | 新增维度 | 输入要求 | 示例 |
|------|----------|----------|------|
| **torch.cat** | 否 | 仅拼接维度大小可不同 | `cat([(2,3), (2,5)], dim=1) → (2,8)` |
| **torch.stack** | 是 | 所有形状必须完全相同 | `stack([(3,4), (3,4)], dim=0) → (2,3,4)` |

## 2. 张量维度操作

### 2.1 unsqueeze() 和 squeeze()

#### unsqueeze() - 增加维度
```python
# 在指定位置增加维度
x = torch.tensor([1, 2, 3])
y = x.unsqueeze(0)  # 在第0维增加维度
print(y.shape)  # torch.Size([1, 3])

z = x.unsqueeze(1)  # 在第1维增加维度
print(z.shape)  # torch.Size([3, 1])
```

#### squeeze() - 移除维度
```python
# 移除大小为1的维度
x = torch.tensor([[[1], [2], [3]]])
print(x.shape)  # torch.Size([1, 3, 1])

y = x.squeeze()  # 移除所有大小为1的维度
print(y.shape)  # torch.Size([3])

z = x.squeeze(0)  # 只移除第0维
print(z.shape)  # torch.Size([3, 1])
```

### 2.2 view() 和 reshape()

#### view() - 改变形状
```python
x = torch.randn(4, 4)
y = x.view(16)  # 展平为一维
z = x.view(2, 8)  # 改变为2x8
w = x.view(-1, 4)  # -1表示自动计算
```

#### reshape() - 重新整形
```python
# reshape() 更灵活，可以处理非连续张量
x = torch.randn(4, 4)
y = x.reshape(16)  # 展平
z = x.reshape(2, 8)  # 改变形状
```

### 2.3 permute() - 维度重排
```python
x = torch.randn(2, 3, 4)
y = x.permute(2, 0, 1)  # 重排维度
print(y.shape)  # torch.Size([4, 2, 3])
```

## 3. 张量数学运算

### 3.1 矩阵乘法

#### torch.matmul() - 矩阵乘法
```python
# 二维矩阵乘法
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.matmul(a, b)  # 等价于 a @ b
print(c.shape)  # torch.Size([3, 5])

# 批量矩阵乘法
a = torch.randn(10, 3, 4)
b = torch.randn(10, 4, 5)
c = torch.matmul(a, b)
print(c.shape)  # torch.Size([10, 3, 5])
```

#### torch.bmm() - 批量矩阵乘法
```python
# 专门用于批量矩阵乘法
a = torch.randn(10, 3, 4)
b = torch.randn(10, 4, 5)
c = torch.bmm(a, b)
print(c.shape)  # torch.Size([10, 3, 5])
```

### 3.2 激活函数

#### torch.tanh() - 双曲正切
```python
x = torch.randn(3, 4)
y = torch.tanh(x)  # 输出范围 [-1, 1]
```

#### torch.sigmoid() - Sigmoid函数
```python
x = torch.randn(3, 4)
y = torch.sigmoid(x)  # 输出范围 (0, 1)
```

## 4. 梯度计算控制

### 4.1 torch.no_grad() - 禁用梯度计算
```python
# 在推理时禁用梯度计算，节省内存
with torch.no_grad():
    output = model(input_data)
    predictions = torch.argmax(output, dim=1)
```

### 4.2 detach() - 分离梯度
```python
# 创建不需要梯度的张量副本
x = torch.randn(3, 4, requires_grad=True)
y = x.detach()  # y不需要梯度
```

## 5. 常见应用场景

### 5.1 特征融合
```python
# 使用cat合并特征
feat1 = torch.randn(32, 64, 56, 56)  # 特征图1
feat2 = torch.randn(32, 64, 56, 56)  # 特征图2
combined = torch.cat([feat1, feat2], dim=1)  # 沿通道维拼接
print(combined.shape)  # torch.Size([32, 128, 56, 56])
```

### 5.2 批量数据处理
```python
# 使用stack构建批量数据
img1 = torch.randn(3, 256, 256)  # 单张图像
img2 = torch.randn(3, 256, 256)
batch = torch.stack([img1, img2], dim=0)  # shape: (2, 3, 256, 256)
```

### 5.3 注意力机制中的QKV处理
```python
# 多头注意力中的维度变换
batch_size, seq_len, d_model = 32, 100, 512
x = torch.randn(batch_size, seq_len, d_model)

# 重塑为多头
num_heads = 8
x = x.view(batch_size, seq_len, num_heads, d_model // num_heads)
x = x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
```

## 6. 性能优化技巧

### 6.1 内存优化
```python
# 使用contiguous()确保内存连续
x = torch.randn(3, 4)
x = x.permute(1, 0)  # 转置后内存可能不连续
x = x.contiguous()   # 确保内存连续，提高访问效率
```

### 6.2 计算优化
```python
# 使用inplace操作节省内存
x = torch.randn(3, 4)
x.add_(1)  # inplace加法，不创建新张量
x.mul_(2)  # inplace乘法
```

## 7. 常见错误与解决方案

### 7.1 形状不匹配错误
```python
# 错误示例
x = torch.randn(2, 3)
y = torch.randn(3, 2)
z = torch.cat([x, y], dim=1)  # 报错！其他维度不一致

# 正确做法
x = torch.randn(2, 3)
y = torch.randn(2, 4)
z = torch.cat([x, y], dim=1)  # 正确
```

### 7.2 梯度计算错误
```python
# 避免在训练时使用no_grad
x = torch.randn(3, 4, requires_grad=True)
with torch.no_grad():
    y = x * 2  # y不会计算梯度
    # 如果需要梯度，不要使用no_grad
```

## 总结

PyTorch张量操作是深度学习的基础，掌握这些操作对于构建和训练模型至关重要：

1. **拼接与堆叠**：`cat()`用于扩展维度，`stack()`用于新增维度
2. **维度操作**：`view()`, `reshape()`, `permute()`等用于改变张量形状
3. **数学运算**：`matmul()`, `bmm()`等用于矩阵运算
4. **梯度控制**：`no_grad()`, `detach()`用于控制梯度计算
5. **性能优化**：合理使用inplace操作和内存连续化

通过熟练掌握这些操作，可以高效地处理各种深度学习任务中的张量数据。 

## 数据加载与处理详解

# PyTorch数据加载与处理详解

## 1. Dataset类详解

### 1.1 自定义Dataset

PyTorch中的Dataset是一个抽象类，需要实现`__len__`和`__getitem__`方法。

#### 基本结构
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

#### 文本数据集示例
```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```

#### 图像数据集示例
```python
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
```

### 1.2 内置Dataset

PyTorch提供了许多预定义的数据集：

```python
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision import transforms

# MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# CIFAR10数据集
cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# 图像文件夹数据集
image_dataset = ImageFolder(root='./data/images', transform=transform)
```

## 2. DataLoader详解

### 2.1 基本用法

`DataLoader`是PyTorch中用于高效加载数据的工具，支持批量加载、多进程加速、数据打乱等功能。

#### 基本配置
```python
from torch.utils.data import DataLoader

# 基本用法
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=False
)

# 迭代使用
for batch_idx, (data, target) in enumerate(dataloader):
    # 训练代码
    pass
```

### 2.2 关键参数详解

| 参数            | 类型        | 默认值     | 说明                 |
| ------------- | --------- | ------- | ------------------ |
| `dataset`     | `Dataset` | -       | 必须传入的数据集对象         |
| `batch_size`  | int       | `1`     | 每个批次的样本数           |
| `shuffle`     | bool      | `False` | 是否在每个epoch打乱数据     |
| `num_workers` | int       | `0`     | 加载数据的子进程数          |
| `drop_last`   | bool      | `False` | 是否丢弃最后不足一个batch的数据 |
| `pin_memory`  | bool      | `False` | 是否将数据复制到CUDA固定内存   |
| `sampler`     | Sampler   | `None`  | 自定义采样策略            |
| `collate_fn`  | callable  | `None`  | 自定义如何合并样本列表为批次     |

### 2.3 高级配置示例

#### 处理变长序列
```python
def collate_fn(batch):
    # 假设batch是[(text, label), ...]，text是变长序列
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 填充文本到最大长度
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return padded_texts, torch.tensor(labels)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True
)
```

#### 多GPU训练配置
```python
from torch.utils.data.distributed import DistributedSampler

# 分布式采样器
sampler = DistributedSampler(dataset, shuffle=True)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    sampler=sampler,  # 使用分布式采样器
    num_workers=4,
    pin_memory=True
)
```

#### 测试集配置
```python
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,  # 测试集不打乱
    num_workers=2,
    drop_last=False  # 保留所有数据
)
```

## 3. 数据预处理与增强

### 3.1 图像预处理

#### 基本变换
```python
import torchvision.transforms as transforms

# 基本图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(           # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

#### 数据增强
```python
# 训练时的数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 颜色抖动
    transforms.RandomRotation(15),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试时的变换（无增强）
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3.2 文本预处理

#### 基本文本处理
```python
from transformers import AutoTokenizer

# 使用预训练模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

def preprocess_text(text):
    # 文本清理
    text = text.strip()
    text = text.lower()
    
    # 分词和编码
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    return encoding
```

## 4. 性能优化技巧

### 4.1 多进程加载
```python
# 根据CPU核心数设置num_workers
import multiprocessing

num_workers = multiprocessing.cpu_count()
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=num_workers,
    pin_memory=True  # GPU训练时启用
)
```

### 4.2 内存优化
```python
# 使用生成器减少内存占用
def data_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

# 或者使用DataLoader的prefetch_factor参数
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2  # 预取2个batch
)
```

### 4.3 缓存机制
```python
class CachedDataset(Dataset):
    def __init__(self, dataset, cache_size=1000):
        self.dataset = dataset
        self.cache = {}
        self.cache_size = cache_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        item = self.dataset[idx]
        
        # 简单的LRU缓存
        if len(self.cache) >= self.cache_size:
            # 移除最旧的项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = item
        return item
```

## 5. 常见问题与解决方案

### 5.1 多进程问题
```python
# Windows或Jupyter中的多进程问题
if __name__ == '__main__':
    # 将DataLoader创建放在这里
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### 5.2 内存不足
```python
# 减少batch_size或num_workers
dataloader = DataLoader(
    dataset,
    batch_size=16,  # 减小batch_size
    num_workers=2,  # 减少worker数量
    pin_memory=False  # 关闭pin_memory
)
```

### 5.3 数据不平衡
```python
from torch.utils.data import WeightedRandomSampler

# 计算样本权重
class_counts = [100, 50, 25]  # 各类别样本数
class_weights = 1.0 / torch.tensor(class_counts)
sample_weights = class_weights[labels]

# 加权采样器
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler
)
```

## 6. 实际应用示例

### 6.1 完整的训练循环
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

# 使用示例
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch}, Average Loss: {train_loss:.4f}')
```

### 6.2 数据验证
```python
def validate_data_loader(dataloader):
    """验证DataLoader是否正确工作"""
    print(f"DataLoader长度: {len(dataloader)}")
    
    for i, batch in enumerate(dataloader):
        if i == 0:  # 只检查第一个batch
            print(f"Batch {i} 形状: {batch[0].shape}")
            print(f"Batch {i} 标签: {batch[1].shape}")
            break
    
    print("数据加载器验证完成")

# 使用示例
validate_data_loader(train_loader)
```

## 总结

PyTorch的数据加载与处理是深度学习项目的基础，掌握这些技巧可以显著提升训练效率：

1. **Dataset设计**：合理设计自定义Dataset类，支持不同类型的数据
2. **DataLoader配置**：根据硬件和任务需求优化参数设置
3. **数据预处理**：使用适当的变换和增强技术
4. **性能优化**：合理使用多进程、缓存等技术
5. **问题解决**：了解常见问题的解决方案

通过熟练掌握这些技术，可以构建高效、稳定的数据加载管道。 

## 训练过程详解

# 训练过程详解

## 1. 深度学习训练的基本流程

深度学习训练的核心是**随机梯度下降（SGD）**，每次迭代包含以下步骤：
1. 读取一小批量训练样本
2. 前向传播获得预测
3. 计算损失
4. 反向传播计算梯度
5. 更新模型参数

## 2. 前向传播（Forward Pass）

### 2.1 基本概念
前向传播是数据从输入层流向输出层的过程，计算图在此过程中动态构建。

### 2.2 线性回归示例
```python
import torch

# 问题设定：学习 y = 2x + 1 + ε
X = torch.rand(100, 1)  # 输入特征 (100 samples)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)  # 真实标签 (带噪声)

# 初始化模型参数
w = torch.randn(1, requires_grad=True)  # 随机初始化权重
b = torch.zeros(1, requires_grad=True)  # 初始偏置

# 前向传播
def forward(x, w, b):
    return w * x + b

# 单次前向传播
y_pred = forward(X, w, b)
print(f"预测值形状: {y_pred.shape}")
print(f"真实值形状: {y.shape}")
```

### 2.3 神经网络前向传播
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 使用示例
model = SimpleNN(input_dim=784, hidden_dim=256, output_dim=10)
x = torch.randn(32, 784)  # batch_size=32
output = model(x)
print(f"输出形状: {output.shape}")  # torch.Size([32, 10])
```

## 3. 损失函数计算

### 3.1 常见损失函数
```python
import torch.nn.functional as F

# 均方误差损失（回归任务）
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

# 交叉熵损失（分类任务）
def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target)

# 二元交叉熵损失（二分类）
def binary_cross_entropy_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

# 示例
y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
y_true = torch.tensor([1, 0])  # 类别标签

ce_loss = cross_entropy_loss(y_pred, y_true)
print(f"交叉熵损失: {ce_loss.item():.4f}")
```

### 3.2 自定义损失函数
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# 使用示例
focal_loss = FocalLoss()
loss = focal_loss(torch.randn(10, 1), torch.randint(0, 2, (10, 1)).float())
```

## 4. 反向传播（Backward Pass）

### 4.1 自动微分原理
PyTorch使用自动微分（Autograd）来计算梯度，通过构建计算图来追踪所有操作。

```python
# 自动微分示例
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(f"dy/dx = {x.grad.item()}")  # 2x + 3 = 7
```

### 4.2 梯度计算过程
```python
# 线性回归的梯度计算
def compute_gradients_manually(x, y, w, b):
    """手动计算梯度（用于理解）"""
    n = len(x)
    
    # 前向传播
    y_pred = w * x + b
    
    # 损失
    loss = ((y_pred - y) ** 2).mean()
    
    # 手动计算梯度
    dw = 2 * ((y_pred - y) * x).mean()
    db = 2 * (y_pred - y).mean()
    
    return dw, db

# 使用PyTorch自动微分
def compute_gradients_autograd(x, y, w, b):
    """使用自动微分计算梯度"""
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    
    return w.grad, b.grad

# 比较两种方法
x = torch.randn(10, 1)
y = 2 * x + 1 + 0.1 * torch.randn(10, 1)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

dw_manual, db_manual = compute_gradients_manually(x, y, w, b)
w.grad.zero_()  # 清零梯度
b.grad.zero_()
dw_auto, db_auto = compute_gradients_autograd(x, y, w, b)

print(f"手动计算梯度: dw={dw_manual:.4f}, db={db_manual:.4f}")
print(f"自动微分梯度: dw={dw_auto.item():.4f}, db={db_auto.item():.4f}")
```

## 5. 参数更新

### 5.1 基本SGD更新
```python
def sgd_update(params, gradients, learning_rate):
    """SGD参数更新"""
    for param, grad in zip(params, gradients):
        param.data -= learning_rate * grad

# 使用示例
learning_rate = 0.01
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 前向传播
y_pred = w * x + b
loss = ((y_pred - y) ** 2).mean()

# 反向传播
loss.backward()

# 参数更新
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()
```

### 5.2 优化器使用
```python
import torch.optim as optim

# 定义模型和优化器
model = SimpleNN(784, 256, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练循环
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
```

## 6. 完整训练循环

### 6.1 线性回归完整示例
```python
def train_linear_regression():
    """线性回归完整训练过程"""
    # 生成数据
    X = torch.rand(100, 1)
    y = 2 * X + 1 + 0.1 * torch.randn(100, 1)
    
    # 初始化参数
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    # 超参数
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 10
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        
        # 批量训练
        for i in range(0, len(X), batch_size):
            # 获取批量数据
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            # 前向传播
            y_pred = w * X_batch + b
            
            # 计算损失
            loss = ((y_pred - y_batch) ** 2).mean()
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
                w.grad.zero_()
                b.grad.zero_()
            
            total_loss += loss.item()
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(X) // batch_size)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    print(f'训练完成! w={w.item():.3f}, b={b.item():.3f}')
    return w, b

# 运行训练
w_trained, b_trained = train_linear_regression()
```

### 6.2 神经网络训练示例
```python
def train_neural_network():
    """神经网络完整训练过程"""
    # 数据准备
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    
    # 模型定义
    model = SimpleNN(784, 256, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 10
    batch_size = 32
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X), batch_size):
            # 获取批量数据
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / (len(X) // batch_size)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model

# 运行训练
trained_model = train_neural_network()
```

## 7. 训练技巧与最佳实践

### 7.1 梯度裁剪
```python
def train_with_gradient_clipping(model, dataloader, max_norm=1.0):
    """使用梯度裁剪的训练"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
```

### 7.2 学习率调度
```python
def train_with_lr_scheduler():
    """使用学习率调度器的训练"""
    model = SimpleNN(784, 256, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        # 训练代码...
        scheduler.step()  # 更新学习率
        
        if epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch}, Learning Rate: {current_lr:.6f}')
```

### 7.3 早停机制
```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# 使用早停
early_stopping = EarlyStopping(patience=5)
for epoch in range(num_epochs):
    # 训练...
    val_loss = validate(model, val_dataloader)
    
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break
```

## 8. 调试与监控

### 8.1 梯度监控
```python
def monitor_gradients(model):
    """监控梯度"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f'{name}: grad_norm = {grad_norm:.4f}')
            
            # 检查梯度爆炸
            if grad_norm > 10:
                print(f'Warning: Large gradient in {name}')
```

### 8.2 损失监控
```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用示例
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练...
    train_loss = train_epoch(model, train_dataloader)
    val_loss = validate_epoch(model, val_dataloader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

plot_training_curves(train_losses, val_losses)
```

## 9. 关键要点总结

1. **前向传播**：数据从输入流向输出，构建计算图
2. **损失计算**：选择合适的损失函数评估模型性能
3. **反向传播**：自动计算梯度，利用链式法则
4. **参数更新**：使用优化器更新模型参数
5. **训练技巧**：梯度裁剪、学习率调度、早停等提高训练稳定性
6. **监控调试**：监控梯度、损失变化，及时发现问题 

