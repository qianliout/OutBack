`torch.utils.data.DataLoader` 是 PyTorch 中用于高效加载数据的工具，它可以将数据集包装成可迭代的对象，支持**批量加载**、**多进程加速**、**数据打乱**等功能，是训练深度学习模型时的核心组件之一。

***

### **作用**

1.  **批量加载**：将数据分成小批量（batches）输入模型。
2.  **多进程加速**：通过多进程预加载数据（`num_workers > 0`），减少训练时的I/O瓶颈。
3.  **数据打乱**：每个 epoch 随机打乱数据（避免模型记忆顺序）。
4.  **自动内存管理**：高效处理大数据集（支持自定义采样策略、内存映射等）。

***

### **基本用法**

```python
from torch.utils.data import DataLoader, Dataset

# 假设已定义自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(data)  # 你的数据集
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代获取数据
for batch in dataloader:
    inputs = batch  # 假设数据已经是张量
    # 训练代码...
```

***

### **关键参数详解**

| 参数              | 类型        | 默认值     | 说明                                                               |
| --------------- | --------- | ------- | ---------------------------------------------------------------- |
| `dataset`       | `Dataset` | -       | 必须传入的数据集对象（需实现 `__len__` 和 `__getitem__`）。                       |
| `batch_size`    | int       | `1`     | 每个批次的样本数。                                                        |
| `shuffle`       | bool      | `False` | 是否在每个 epoch 打乱数据（训练集通常设为 `True`，测试集为 `False`）。                   |
| `num_workers`   | int       | `0`     | 加载数据的子进程数（多进程加速，Windows下可能需用 `if __name__ == '__main__'` 保护主代码）。 |
| `drop_last`     | bool      | `False` | 是否丢弃最后不足一个 batch 的数据（避免尺寸不匹配）。                                   |
| `pin_memory`    | bool      | `False` | 是否将数据复制到 CUDA 固定内存（加速 GPU 传输）。                                   |
| `sampler`       | Sampler   | `None`  | 自定义采样策略（如分布式训练用 `DistributedSampler`）。                           |
| `batch_sampler` | Sampler   | `None`  | 自定义批量采样（覆盖 `batch_size` 和 `shuffle`）。                            |
| `collate_fn`    | callable  | `None`  | 自定义如何合并样本列表为批次（处理不定长数据时常用）。                                      |

***

### **常用场景示例**

#### 1. **处理不定长数据（如文本）**

通过 `collate_fn` 动态填充：

```python
def collate_fn(batch):
    # 假设 batch 是 [(text, label), ...]，text 是变长序列
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # 填充文本到最大长度
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return padded_texts, torch.tensor(labels)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

#### 2. **多GPU训练配合 `DistributedSampler`**

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### 3. **测试集配置**

```python
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

***

### **注意事项**

1.  **多进程问题**：在 Windows 或 Jupyter 中，多进程可能导致错误，需将代码放在 `if __name__ == '__main__':` 中。
2.  **内存限制**：`num_workers` 过大可能耗尽内存，建议根据 CPU 核心数调整（通常设为 `4` 或 `8`）。
3.  **GPU 加速**：设置 `pin_memory=True` 可提升 GPU 数据传输效率（需配合 `.to('cuda')` 使用）。

通过合理配置 `DataLoader`，可以显著提升模型训练效率，尤其在大规模数据集上。
