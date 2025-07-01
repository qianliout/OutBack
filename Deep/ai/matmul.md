## **`torch.matmul` 的用法详解**

`torch.matmul` 是 PyTorch 中用于执行**矩阵乘法**或**张量乘法**的函数，支持多种维度的张量运算。它比 `torch.mm`（仅限二维矩阵乘法）和 `@` 运算符更灵活，适用于高维张量的批量矩阵乘法。

***

## **1. 核心功能**

*   **矩阵乘法**（2D 张量）：`(m×n) @ (n×p) → (m×p)`
*   **批量矩阵乘法**（3D 张量）：`(b×m×n) @ (b×n×p) → (b×m×p)`
*   **广播机制**：自动扩展维度以匹配运算要求（类似 NumPy）。
*   **向量-矩阵乘法**、**向量点积**等。

***

## **2. 基本用法**

## **（1）二维矩阵乘法（等价于 `torch.mm`）**

```python
import torch

A = torch.randn(2, 3)  # 形状 (2, 3)
B = torch.randn(3, 4)  # 形状 (3, 4)
C = torch.matmul(A, B)  # 结果形状 (2, 4)
print(C.shape)  # torch.Size([2, 4])
```

## **（2）批量矩阵乘法（3D 张量）**

```python
A = torch.randn(5, 2, 3)  # 批量大小=5，每个矩阵形状 (2, 3)
B = torch.randn(5, 3, 4)  # 批量大小=5，每个矩阵形状 (3, 4)
C = torch.matmul(A, B)  # 结果形状 (5, 2, 4)
print(C.shape)  # torch.Size([5, 2, 4])
```

## **（3）向量与矩阵乘法**

```python
v = torch.randn(3)      # 形状 (3,)
M = torch.randn(3, 4)   # 形状 (3, 4)
result = torch.matmul(v, M)  # 结果形状 (4,)
print(result.shape)  # torch.Size([4])
```

***

## **3. 广播机制（Broadcasting）**

`torch.matmul` 支持广播规则，自动扩展维度以匹配运算：

```python
A = torch.randn(2, 3)    # 形状 (2, 3)
B = torch.randn(5, 3, 4) # 形状 (5, 3, 4)
C = torch.matmul(A, B)   # A 广播为 (5, 2, 3)，结果形状 (5, 2, 4)
print(C.shape)  # torch.Size([5, 2, 4])
```

***

## **4. 不同维度的张量乘法**

| 输入维度         | 运算规则       | 示例                            |
| ------------ | ---------- | ----------------------------- |
| **1D @ 1D**  | 向量点积（返回标量） | `(n,) @ (n,) → scalar`        |
| **1D @ 2D**  | 向量-矩阵乘法    | `(n,) @ (n,m) → (m,)`         |
| **2D @ 1D**  | 矩阵-向量乘法    | `(m,n) @ (n,) → (m,)`         |
| **2D @ 2D**  | 标准矩阵乘法     | `(m,n) @ (n,p) → (m,p)`       |
| **3D @ 3D**  | 批量矩阵乘法     | `(b,m,n) @ (b,n,p) → (b,m,p)` |
| **混合维度（广播）** | 自动扩展维度     | `(m,n) @ (b,n,p) → (b,m,p)`   |

***

## **5. 常见错误及解决方法**

## **错误 1：维度不匹配**

```python
A = torch.randn(2, 3)
B = torch.randn(4, 5)  # 不匹配的维度
C = torch.matmul(A, B)  # 报错：Shape mismatch for matrix multiplication
```

**解决**：确保 `A` 的最后一维与 `B` 的倒数第二维相同（`A.shape[-1] == B.shape[-2]`）。

## **错误 2：标量或高维不适用**

```python
A = torch.randn(2)
B = torch.randn(2)
C = torch.matmul(A, B)  # 返回标量（点积），可能不符合预期
```

**解决**：如需外积（结果矩阵），先扩展维度：

```python
A = A.unsqueeze(1)  # (2) → (2, 1)
B = B.unsqueeze(0)  # (2) → (1, 2)
C = torch.matmul(A, B)  # 结果形状 (2, 2)
```

***

## **6. 与相关函数的对比**

| 函数                 | 适用场景                    | 限制       |
| ------------------ | ----------------------- | -------- |
| **`torch.matmul`** | 通用矩阵/张量乘法（推荐）           | 无        |
| **`torch.mm`**     | 仅限二维矩阵乘法                | 不支持广播    |
| **`torch.bmm`**    | 严格的批量矩阵乘法（3D 张量）        | 输入必须为 3D |
| **`@` 运算符**        | 语法糖（等价于 `torch.matmul`） | 无        |

***

## **7. 实际应用示例**

## **（1）神经网络的全连接层**

```python
# 模拟全连接层：输入 (batch, 512)，权重 (512, 256)
x = torch.randn(32, 512)  # 批量大小=32
W = torch.randn(512, 256)
b = torch.randn(256)
output = torch.matmul(x, W) + b  # 结果形状 (32, 256)
```

## **（2）注意力机制（Q\@K^T）**

```python
Q = torch.randn(5, 10, 64)  # (batch, seq_len, dim)
K = torch.randn(5, 20, 64)  # (batch, seq_len, dim)
scores = torch.matmul(Q, K.transpose(1, 2))  # 结果形状 (5, 10, 20)
```

***

## **8. 总结**

*   **核心功能**：执行矩阵乘法，支持广播和批量运算。
*   **推荐场景**：
    *   神经网络中的线性变换（如全连接层、注意力机制）。
    *   高维张量的批量运算。
*   **替代方案**：
    *   `@` 运算符：代码更简洁（如 `x @ W`）。
    *   `torch.bmm`：仅用于严格的 3D 张量乘法。

**代码模板**：

```python
output = torch.matmul(tensor1, tensor2)  # 通用推荐
output = tensor1 @ tensor2              # 简洁写法（功能相同）
```

