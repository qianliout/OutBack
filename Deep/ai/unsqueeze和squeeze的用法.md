在PyTorch中，`unsqueeze()`是一个非常重要的张量形状操作函数，用于在指定维度上增加一个长度为1的维度。具体到您代码中的这行：

```python
features = queries.unsqueeze(2) + keys.unsqueeze(1)
```

### 核心功能

1.  **`unsqueeze(dim)`**
    *   在指定的`dim`维度位置插入一个大小为1的新维度
    *   不改变张量的数据，只改变形状

### 在注意力机制中的应用

这段代码实现了加性注意力机制的关键步骤：

1.  **`queries.unsqueeze(2)`**
    *   假设原始`queries`形状为 `(batch_size, num_queries, num_hiddens)`
    *   在维度2（从0开始计数）插入新维度 → `(batch_size, num_queries, 1, num_hiddens)`

2.  **`keys.unsqueeze(1)`**
    *   假设原始`keys`形状为 `(batch_size, num_kv_pairs, num_hiddens)`
    *   在维度1插入新维度 → `(batch_size, 1, num_kv_pairs, num_hiddens)`

3.  **广播相加**
    *   通过unsqueeze调整形状后，两个张量会自动广播(broadcast)为相同的形状 `(batch_size, num_queries, num_kv_pairs, num_hiddens)`
    *   然后进行逐元素相加，得到注意力特征

### 示例说明

假设：

*   `queries`: (2, 3, 8)
*   `keys`: (2, 5, 8)

操作后：

*   `queries.unsqueeze(2)`: (2, 3, 1, 8)
*   `keys.unsqueeze(1)`: (2, 1, 5, 8)
*   相加结果: (2, 3, 5, 8)

### 为什么需要这样做？

这种形状变换使得：

1.  每个查询可以与所有键值对进行计算
2.  保持了批处理维度的独立性
3.  为后续的注意力权重计算准备了合适的张量形状

## squeeze

在PyTorch中，`torch.squeeze()` 是一个常用的张量操作函数，用于**移除张量中维度大小为1的轴**（即“压缩”张量），从而降低张量的维度。它的主要用途是处理那些因为某些操作（如`unsqueeze`或卷积）而多出的冗余单维度。

***

## **1. 基本语法**

```python
torch.squeeze(input, dim=None) → Tensor
```

*   **`input`**: 输入张量。
*   **`dim`** (可选): 指定要移除的维度。如果未指定，默认移除所有大小为1的维度。

***

## **2. 使用示例**

### **示例1：默认移除所有单维度**

```python
import torch

x = torch.randn(1, 3, 1, 2)  # 形状: [1, 3, 1, 2]
y = torch.squeeze(x)          # 移除所有大小为1的维度
print(y.shape)                # 输出: torch.Size([3, 2])
```

**解释**：\
原始张量 `x` 的形状是 `[1, 3, 1, 2]`，其中第0维和第2维的大小为1。调用 `squeeze()` 后，这两个维度被移除，结果形状变为 `[3, 2]`。

***

### **示例2：指定移除某一单维度**

```python
x = torch.randn(1, 3, 1, 2)  # 形状: [1, 3, 1, 2]
y = torch.squeeze(x, dim=0)   # 仅移除第0维
print(y.shape)                # 输出: torch.Size([3, 1, 2])

z = torch.squeeze(x, dim=2)   # 仅移除第2维
print(z.shape)                # 输出: torch.Size([1, 3, 2])
```

**关键点**：

*   如果 `dim` 指定的维度大小不为1，则张量形状**不会改变**（不会报错）。
*   例如，`torch.squeeze(x, dim=1)` 对 `x` 无效，因为第1维的大小是3。

***

### **示例3：处理无单维度的张量**

```python
x = torch.randn(2, 3)        # 形状: [2, 3]
y = torch.squeeze(x)         # 无单维度，形状不变
print(y.shape)               # 输出: torch.Size([2, 3])
```

***

## **3. 常见用途**

### **(1) 移除批量维度（当batch\_size=1时）**

```python
x = torch.randn(1, 64, 64)   # 形状: [1, 64, 64] (单张图像)
y = torch.squeeze(x, dim=0)  # 移除第0维
print(y.shape)               # 输出: torch.Size([64, 64])
```

### **(2) 与`unsqueeze`配合使用**

```python
x = torch.randn(3, 2)        # 形状: [3, 2]
y = x.unsqueeze(dim=1)       # 形状: [3, 1, 2]
z = torch.squeeze(y, dim=1)  # 恢复原始形状
print(z.shape)               # 输出: torch.Size([3, 2])
```

### **(3) 处理模型输出**

某些模型（如LSTM）的输出可能包含单维度：

```python
output, _ = lstm(input_data)  # output形状: [batch_size, seq_len, 1, hidden_dim]
output = output.squeeze(dim=2)  # 形状: [batch_size, seq_len, hidden_dim]
```

***

## **4. 与`numpy.squeeze`的对比**

PyTorch的`squeeze`和NumPy的`squeeze`功能一致，但语法略有不同：

```python
# PyTorch
y = torch.squeeze(x, dim=2)

# NumPy
y = np.squeeze(x, axis=2)
```

***

## **5. 注意事项**

1.  **非单维度不可压缩**：\
    如果指定 `dim` 的维度大小不为1，则张量形状不变（不会报错）。
    ```python
    x = torch.randn(2, 3)
    y = torch.squeeze(x, dim=1)  # 无效，因为dim=1的大小是3
    print(y.shape)               # 输出: torch.Size([2, 3])
    ```

2.  **原地操作**：\
    `squeeze` 默认返回新张量，若想原地修改，需使用 `squeeze_`：
    ```python
    x.squeeze_(dim=0)  # 直接修改x
    ```

***

## **6. 总结**

| 场景      | 代码                        | 作用             |
| ------- | ------------------------- | -------------- |
| 移除所有单维度 | `torch.squeeze(x)`        | 压缩所有大小为1的维度    |
| 移除指定单维度 | `torch.squeeze(x, dim=2)` | 仅压缩第2维（如果大小为1） |
| 检查是否可压缩 | `x.shape`                 | 确保目标维度大小为1     |

**一句话总结**：\
`torch.squeeze()` 是“降维工具”，用于清理张量中不必要的单维度，尤其在处理模型输入/输出时非常有用！
