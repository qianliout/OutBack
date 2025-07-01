# &#x20;用法

在PyTorch中，`torch.stack()` 是一个用于**将多个张量沿新维度堆叠**的函数。与 `torch.cat()` 不同，`stack()` 会创建一个新的维度，而 `cat()` 是在现有维度上扩展。以下是其详细用法及参数说明：

***

### **1. 函数定义**

```python
torch.stack(tensors, dim=0, *, out=None) → Tensor
```

*   **参数**：
    *   `tensors` (sequence of Tensors): 需要堆叠的张量序列（如列表或元组）。
    *   `dim` (int, optional): 指定新维度的位置（默认`dim=0`）。
    *   `out` (Tensor, optional): 可选输出张量（一般不手动指定）。
*   **返回值**：堆叠后的新张量。

***

### **2. 核心规则**

1.  **所有张量的形状必须完全相同**（包括每个维度的大小）。
2.  **会新增一个维度**，新维度的大小等于输入张量的数量。
3.  **堆叠后的张量维度数 = 输入张量维度数 + 1**。

***

### **3. 参数详解与示例**

#### **(1) `tensors`：待堆叠的张量序列**

*   所有张量的 `shape` 必须完全一致。
*   可以是同类型的任意维度张量（如标量、向量、矩阵等）。

#### **(2) `dim`：指定新维度的位置**

*   **`dim=0`**：在最外层新增维度（默认）。
*   **`dim=1`**：在第1维新增维度。
*   **`dim=-1`**：在最后一维新增维度。

***

### **4. 具体示例**

#### **示例1：堆叠两个标量（新增第0维）**

```python
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

z = torch.stack([x, y])  # dim=0
print(z)  # tensor([1., 2.])
print(z.shape)  # torch.Size([2])
```

**解释**：将两个标量堆叠为一个长度为2的向量。

#### **示例2：堆叠两个向量（新增第0维）**

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

z = torch.stack([x, y])  # dim=0
print(z)
```

输出：

    tensor([[1, 2, 3],
            [4, 5, 6]])

**形状变化**：`(3,)` + `(3,)` → `(2, 3)`。

#### **示例3：指定 `dim=1` 新增维度**

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

z = torch.stack([x, y], dim=1)
print(z)
```

输出：

    tensor([[1, 4],
            [2, 5],
            [3, 6]])

**形状变化**：`(3,)` + `(3,)` → `(3, 2)`。

#### **示例4：堆叠矩阵（新增第3维）**

```python
x = torch.randn(2, 3)  # shape: (2, 3)
y = torch.randn(2, 3)

z = torch.stack([x, y], dim=2)  # 新增第3维
print(z.shape)  # torch.Size([2, 3, 2])
```

***

### **5. 常见用途**

1.  **合并多个损失值**
    ```python
    loss1 = torch.tensor(0.5)
    loss2 = torch.tensor(0.8)
    combined_loss = torch.stack([loss1, loss2])  # tensor([0.5, 0.8])
    ```
2.  **构建批量数据**
    ```python
    img1 = torch.randn(3, 256, 256)  # 单张图像
    img2 = torch.randn(3, 256, 256)
    batch = torch.stack([img1, img2], dim=0)  # shape: (2, 3, 256, 256)
    ```
3.  **时间序列数据堆叠**
    ```python
    seq1 = torch.randn(10, 5)  # 10个时间步的特征
    seq2 = torch.randn(10, 5)
    stacked_seq = torch.stack([seq1, seq2], dim=1)  # shape: (10, 2, 5)
    ```

***

### **6. 与 `torch.cat` 的区别**

| 方法                | 是否新增维度 | 输入形状要求     | 示例                                       |
| ----------------- | ------ | ---------- | ---------------------------------------- |
| **`torch.stack`** | 是      | 所有张量形状必须相同 | `stack([(3,4), (3,4)], dim=0) → (2,3,4)` |
| **`torch.cat`**   | 否      | 仅拼接维度大小可不同 | `cat([(2,3), (2,5)], dim=1) → (2,8)`     |

***

### **7. 易错点**

1.  **形状不一致**：
    ```python
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    z = torch.stack([x, y])  # 报错！形状不匹配
    ```
2.  **空张量列表**：\
    `torch.stack([])` 会报错（至少需要一个张量）。

***

### **8. 总结**

*   **核心功能**：通过新增维度堆叠多个张量。
*   **关键参数**：
    *   `tensors`：待堆叠的张量列表（形状必须相同）。
    *   `dim`：指定新维度的位置（支持负数索引）。
*   **典型场景**：构建批量数据、合并同形状张量、时间序列处理。

通过合理使用 `torch.stack()`，可以高效组织张量数据！
