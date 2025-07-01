```python
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
```

这段代码使用了PyTorch张量的`repeat()`方法来实现张量的扩展广播(broadcasting)操作。具体解释如下：

1.  **功能作用**：
    *   将形状为`(batch_size, num_hiddens)`的`context`张量
    *   扩展为`(num_steps, batch_size, num_hiddens)`的形状
    *   使每个时间步都能使用相同的上下文向量

2.  **参数解析**：
    *   `embs.shape[0]`：获取embs张量的第0维大小（即时间步数`num_steps`）
    *   `repeat(embs.shape[0], 1, 1)`表示：
        *   在第0维（时间步维度）复制`num_steps`次
        *   第1维（batch维度）保持不变（参数1）
        *   第2维（隐藏层维度）保持不变（参数1）

3.  **实际效果**：
    ```python
    # 假设原始context形状为(64, 256) (batch_size=64, num_hiddens=256)
    # embs.shape[0]=10 (num_steps=10)
    context = context.repeat(10, 1, 1) 
    # 结果形状变为(10, 64, 256)
    ```

4.  **在Seq2Seq模型中的意义**：
    *   使编码器输出的最终隐藏状态能够与解码器所有时间步的输入拼接
    *   保持了批次和时间步的对应关系
    *   为后续的`torch.cat`操作准备匹配的形状

这种操作是序列模型中常见的技巧，用于将单步状态信息扩展到整个序列长度。

## repeat方法的用法

在PyTorch中，`Tensor.repeat()` 方法用于**沿指定维度重复复制张量的数据**，生成一个更大的新张量。与 `expand()` 不同，`repeat()` 会真实复制数据（而非共享内存）。以下是其详细说明：

***

### **1. 方法定义**

```python
Tensor.repeat(*sizes) → Tensor
```

*   **参数**：\
    `*sizes` 是一个可变参数，表示每个维度需要重复的次数。
*   **返回值**：\
    返回一个新的张量，原始数据按指定次数在各个维度上复制。

***

### **2. 参数意义与规则**

#### **(1) 参数 `*sizes`**

*   **长度要求**：\
    `sizes` 的参数个数可以 ≥ 张量的维度数（`dim()`）。
    *   如果 `sizes` 长度 > 张量维度，PyTorch 会自动在张量前面补 `1` 个维度。
    *   如果 `sizes` 长度 < 张量维度，会从最左侧维度开始匹配。

*   **每个参数的含义**：\
    `sizes[i]` 指定第 `i` 个维度（从外到内）的重复次数。
    *   例如 `x.repeat(2, 3)` 表示：
        *   第0维（行）重复2次。
        *   第1维（列）重复3次。

#### **(2) 维度扩展规则**

假设原始张量形状为 `(d0, d1, ..., dn)`，`repeat(s0, s1, ..., sm)` 的输出形状为：

*   若 `m > n`，先在原始张量前补 `1` 维，变为 `(1, ..., 1, d0, d1, ..., dn)`，再重复。
*   输出形状的每个维度大小为：`原始大小 * sizes[i]`。

***

### **3. 具体示例**

#### **示例1：1D张量（向量）**

```python
import torch

x = torch.tensor([1, 2, 3])  # shape: (3,)

# 第0维重复2次
y = x.repeat(2)  # 形状: (6,)
print(y)  # tensor([1, 2, 3, 1, 2, 3])

# 补1维后重复（等效于先unsqueeze(0)）
z = x.repeat(2, 3)  # 视为 (1,3) -> (2,9)
print(z)
```

输出：

    tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3]])

#### **示例2：2D张量（矩阵）**

```python
x = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)

# 第0维重复2次，第1维重复3次
y = x.repeat(2, 3)  # 形状: (4, 6)
print(y)
```

输出：

    tensor([[1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]])

#### **示例3：高维张量（自动补1维）**

```python
x = torch.randn(3, 2)  # shape: (3, 2)

# sizes长度 > 张量维度时，前面补1维
y = x.repeat(2, 1, 3)  # 视为 (1,3,2) -> (2,3,6)
print(y.shape)  # torch.Size([2, 3, 6])
```

***

### **4. 关键点总结**

| 参数/行为                | 说明                                    |
| -------------------- | ------------------------------------- |
| **`*sizes`**         | 每个参数对应维度的重复次数，长度可≥张量维度。               |
| **自动补1维**            | 若 `sizes` 比张量维度长，左侧补1维后再重复。           |
| **输出形状**             | 每个维度大小 = 原始大小 × `sizes[i]`。           |
| **与 `expand()` 的区别** | `repeat()` 复制数据，`expand()` 是视图（共享内存）。 |

***

### **5. 常见用途**

1.  **批量数据扩展**
    ```python
    x = torch.randn(10, 3)  # 单个样本
    batch_x = x.repeat(32, 1)  # 扩展为32个样本 (32,10,3)
    ```
2.  **图像拼贴**
    ```python
    patch = torch.randn(16, 16, 3)  # 图像块
    tiled = patch.repeat(4, 4, 1)   # 拼贴为4x4网格 (64,64,3)
    ```
3.  **广播替代**\
    当 `expand()` 不满足需求时（如需要真实复制数据）：
    ```python
    x = torch.tensor([1, 2, 3])
    y = x.repeat(3, 1)  # (3,3)，而非广播视图
    ```

***

### **6. 易错点**

*   **负数参数**：`repeat(-1)` 会报错（`numpy.tile()` 允许负数）。
*   **内存占用**：`repeat()` 会真实复制数据，大张量需谨慎使用。

通过合理使用 `repeat()`，可以高效实现张量的维度扩展！
