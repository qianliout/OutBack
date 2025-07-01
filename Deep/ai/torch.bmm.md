在PyTorch中，`torch.bmm`（**Batch Matrix Multiplication**）用于执行**批量矩阵乘法**，适用于三维张量（批量数据中的矩阵相乘）。以下是详细解释和具体示例：

***

### **1. 函数定义**

```python
torch.bmm(input, mat2, *, out=None) → Tensor
```

*   **输入**：
    *   `input`：形状为 `(batch_size, n, m)` 的三维张量。
    *   `mat2`：形状为 `(batch_size, m, p)` 的三维张量。
*   **输出**：
    *   形状为 `(batch_size, n, p)` 的三维张量。

***

### **2. 核心规则**

1.  **批量维度一致**：`input` 和 `mat2` 的第0维（`batch_size`）必须相同。
2.  **矩阵乘法规则**：每个样本的 `(n, m)` 矩阵与 `(m, p)` 矩阵相乘，得到 `(n, p)` 矩阵。
3.  **不支持广播**：必须显式指定批量维度。

***

### **3. 示例验证**

#### **给定数据**

```python
import torch

# 定义输入张量
a = torch.randn(2, 1, 10)  # shape: (batch_size=2, n=1, m=10)
b = torch.randn(2, 10, 4)  # shape: (batch_size=2, m=10, p=4)

# 执行批量矩阵乘法
result = torch.bmm(a, b)
print(result.shape)  # 输出形状
```

**输出**：

    torch.Size([2, 1, 4])

#### **计算过程分解**

1.  **样本1**：
    *   `a[0]` 形状：`(1, 10)`
    *   `b[0]` 形状：`(10, 4)`
    *   矩阵乘法：`(1,10) @ (10,4) → (1,4)`

2.  **样本2**：
    *   `a[1]` 形状：`(1, 10)`
    *   `b[1]` 形状：`(10, 4)`
    *   矩阵乘法：`(1,10) @ (10,4) → (1,4)`

3.  **最终结果**：
    *   将两个 `(1,4)` 的结果堆叠，得到 `(2,1,4)`。

***

### **4. 数学表示**

对于批量中的每个样本 ( i )：

```
$$
\text{result}\[i] = \text{input}\[i] \times \text{mat2}\[i]
$$
```

*   ( \text{input}\[i] ) 形状：( (n, m) )
*   ( \text{mat2}\[i] ) 形状：( (m, p) )
*   ( \text{result}\[i] ) 形状：( (n, p) )

***

### **5. 与 `torch.matmul` 的区别**

| 方法             | 输入维度                         | 广播支持 | 适用场景        |
| -------------- | ---------------------------- | ---- | ----------- |
| `torch.bmm`    | 严格 `(b, n, m)` 和 `(b, m, p)` | 否    | 明确的批量矩阵乘法   |
| `torch.matmul` | 支持高维广播                       | 是    | 通用矩阵乘法（含批量） |

**示例**：

```python
# 使用 matmul 实现相同效果
result_matmul = torch.matmul(a, b)  # 输出形状同样为 (2,1,4)
```

***

### **6. 常见用途**

1.  **注意力机制中的权重计算**：
    ```python
    # query: (batch_size, seq_len_q, d_k)
    # key:   (batch_size, seq_len_k, d_k)
    scores = torch.bmm(query, key.transpose(1, 2))
    ```
2.  **批量线性变换**：
    ```python
    # x: (batch_size, num_points, in_features)
    # w: (batch_size, in_features, out_features)
    out = torch.bmm(x, w)
    ```

***

### **7. 易错点**

1.  **维度不匹配**：
    ```python
    a = torch.randn(2, 1, 10)
    b = torch.randn(3, 10, 4)  # 报错！batch_size 不一致
    torch.bmm(a, b)
    ```
2.  **错误维度顺序**：
    ```python
    a = torch.randn(2, 10, 1)  # 错误！应为 (2,1,10)
    b = torch.randn(2, 10, 4)
    torch.bmm(a, b)  # 报错！(10,1) 无法与 (10,4) 相乘
    ```

***

### **总结**

*   **输入形状**：`a=(2,1,10)`, `b=(2,10,4)`。
*   **输出形状**：`(2,1,4)`。
*   **关键点**：逐样本矩阵乘法，批量维度必须一致。

通过 `torch.bmm` 可以高效处理批量矩阵运算，尤其在Transformer等模型中广泛应用！
