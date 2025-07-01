### **PyTorch 中的 `nn.GRU` 详解**

`nn.GRU` 是 PyTorch 中实现的门控循环单元（Gated Recurrent Unit），用于处理序列数据（如时间序列、文本、语音）。相比标准 RNN，GRU 通过**门控机制**缓解梯度消失问题，能更有效地建模长距离依赖。

***

## **1. `nn.GRU` 的核心作用**

*   **输入**：序列数据（如句子、时间步信号）。
*   **输出**：每个时间步的隐状态（hidden state），可进一步用于分类、生成等任务。
*   **特点**：
    *   通过\*\*更新门（Update Gate）**和**重置门（Reset Gate）\*\*控制信息流动。
    *   比 LSTM 更简单（少一个门），计算效率更高。

***

## **2. 类的定义**

```python
torch.nn.GRU(
    input_size,          # 输入特征的维度
    hidden_size,         # 隐状态的维度
    num_layers=1,        # GRU 的层数（默认1）
    bias=True,           # 是否使用偏置项
    batch_first=False,   # 输入/输出的形状是否为 (batch, seq, feature)
    dropout=0.0,         # 层间dropout概率（非最后一层）
    bidirectional=False, # 是否双向GRU
)
```

***

## **3. 参数含义**

| 参数名             | 类型    | 说明                                                                  |
| --------------- | ----- | ------------------------------------------------------------------- |
| `input_size`    | int   | 输入特征的维度（如词嵌入维度）。                                                    |
| `hidden_size`   | int   | 隐状态的维度（决定输出向量的长度）。                                                  |
| `num_layers`    | int   | GRU 的堆叠层数（层数 >1 时为深度GRU）。                                           |
| `bias`          | bool  | 是否在门控计算中添加偏置项（默认 `True`）。                                           |
| `batch_first`   | bool  | 若为 `True`，输入形状为 `(batch, seq, feature)`；否则 `(seq, batch, feature)`。 |
| `dropout`       | float | 层间 dropout 概率（仅当 `num_layers >1` 时生效）。                              |
| `bidirectional` | bool  | 若为 `True`，使用双向GRU（合并前后向隐状态）。                                        |

***

## **4. 输入输出形状**

### **输入 `input`**

*   若 `batch_first=False`（默认）：\
    `(seq_len, batch, input_size)`
    *   `seq_len`：序列长度（如句子词数）。
    *   `batch`：批量大小。
    *   `input_size`：输入特征维度。
*   若 `batch_first=True`：\
    `(batch, seq_len, input_size)`

### **输出 `output`, `h_n`**

1.  **`output`**：所有时间步的隐状态
    *   形状：
        *   单向GRU：`(seq_len, batch, hidden_size)`
        *   双向GRU：`(seq_len, batch, 2*hidden_size)`
    *   用途：通常用于序列标注或注意力机制。

2.  **`h_n`**：最后一个时间步的隐状态
    *   形状：
        *   单向GRU：`(num_layers, batch, hidden_size)`
        *   双向GRU：`(num_layers*2, batch, hidden_size)`
    *   用途：用于分类任务或初始化下一个序列。

***

## **5. 前向传播方法**

```python
output, h_n = gru(input, h_0=None)
```

*   **`input`**：输入序列（形状见上文）。
*   **`h_0`**：初始隐状态（默认为零）。\
    形状：`(num_layers * num_directions, batch, hidden_size)`
    *   `num_directions`：2（双向）或1（单向）。

***

## **6. GRU 的数学公式**

GRU 通过以下门控机制更新隐状态：

```math
\begin{aligned}
z\_t &= \sigma(W\_{iz} x\_t + W\_{hz} h\_{t-1} + b\_z) \quad &\text{(更新门)} \\
r\_t &= \sigma(W\_{ir} x\_t + W\_{hr} h\_{t-1} + b\_r) \quad &\text{(重置门)} \\
n\_t &= \tanh(W\_{in} x\_t + r\_t \odot (W\_{hn} h\_{t-1}) + b\_n) \quad &\text{(候选隐状态)} \\
h\_t &= (1 - z\_t) \odot h\_{t-1} + z\_t \odot n\_t \quad &\text{(最终隐状态)}
\end{aligned}
```

*   ( `$\sigma$` )：Sigmoid 函数。
*   ( `$\odot$` )：逐元素乘法。
*   ( `$z\_t$` )：控制新旧隐状态的混合比例。
*   ( `$r\_t$` )：控制历史信息的遗忘程度。

***

## **7. 使用示例**

### **(1) 单向单层 GRU**

```python
import torch
import torch.nn as nn

# 定义GRU：输入维度=10，隐状态维度=20
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1)

# 输入：序列长度=5，批量大小=3，特征维度=10
input = torch.randn(5, 3, 10)  # (seq_len, batch, input_size)

# 前向传播
output, h_n = gru(input)
print(output.shape)  # torch.Size([5, 3, 20])
print(h_n.shape)     # torch.Size([1, 3, 20])
```

### **(2) 双向双层 GRU**

```python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
output, h_n = gru(input)
print(output.shape)  # torch.Size([5, 3, 40])  # 前后向隐状态拼接
print(h_n.shape)     # torch.Size([4, 3, 20])  # 2层x2方向=4
```

### **(3) 使用 `batch_first`**

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
input = torch.randn(3, 5, 10)  # (batch, seq_len, input_size)
output, h_n = gru(input)
print(output.shape)  # torch.Size([3, 5, 20])
```

***

## **8. 注意事项**

1.  **梯度问题**：
    *   GRU 仍可能面临梯度消失/爆炸，但比标准 RNN 更稳定。
    *   可配合梯度裁剪（`torch.nn.utils.clip_grad_norm_`）使用。

2.  **初始化隐状态**：
    *   默认 `h_0` 为零张量，也可自定义：
        ```python
        h_0 = torch.randn(2, 3, 20)  # (num_layers, batch, hidden_size)
        ```

3.  **与 LSTM 的区别**：
    *   GRU 少一个门（无输出门），参数更少，计算更快。
    *   在多数任务中，GRU 和 LSTM 性能接近。

***

## **9. 总结**

| **组件**          | **说明**                                                          |
| --------------- | --------------------------------------------------------------- |
| **输入形状**        | `(seq_len, batch, input_size)` 或 `(batch, seq_len, input_size)` |
| **输出 `output`** | 所有时间步的隐状态（用于序列任务）                                               |
| **输出 `h_n`**    | 最后一个时间步的隐状态（用于分类或初始化下一序列）                                       |
| **核心门控**        | 更新门（`z_t`）和重置门（`r_t`）                                           |
| **应用场景**        | 机器翻译、文本生成、时间序列预测等需要建模长依赖的任务。                                    |

通过合理配置 `nn.GRU` 的参数，可以高效处理序列数据并捕捉长期依赖关系。
