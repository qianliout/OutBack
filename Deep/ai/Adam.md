### **PyTorch 中的 Adam 优化器详解**

Adam（Adaptive Moment Estimation）是一种结合了 **动量（Momentum）** 和 **自适应学习率（RMSProp）** 的优化算法，广泛用于深度学习训练。以下是其在 PyTorch 中的实现原理、参数解释和实际用法。

***

## **1. Adam 的数学原理**

Adam 通过计算梯度的一阶矩（均值）和二阶矩（未中心化的方差）来动态调整每个参数的学习率。

### **更新步骤**

对于每个参数 ( \theta\_t )：

1.  **计算梯度**：

```math
    g\_t = \nabla\_\theta J(\theta\_t)
```

1.  **更新一阶矩（动量）**：

```math
    m\_t = \beta\_1 m\_{t-1} + (1 - \beta\_1) g\_t
```

1.  **更新二阶矩（自适应学习率）**：

```math
    v\_t = \beta\_2 v\_{t-1} + (1 - \beta\_2) g\_t^2
```

1.  **偏差校正**（解决初始零偏差）：

```math
    \hat{m}\_t = \frac{m\_t}{1 - \beta\_1^t}, \quad \hat{v}\_t = \frac{v\_t}{1 - \beta\_2^t}
```

1.  **参数更新**：

```math
    \theta\_{t+1} = \theta\_t - \eta \cdot \frac{\hat{m}\_t}{\sqrt{\hat{v}\_t} + \epsilon}
```

`$\eta$`：学习率（`lr`）。
`$\epsilon$`：数值稳定项（`eps`）。

***

## **2. PyTorch 中的 `torch.optim.Adam`**

### **初始化参数**

```python
optimizer = torch.optim.Adam(
    params,                # 待优化的模型参数（如 model.parameters()）
    lr=1e-3,               # 学习率（默认 0.001）
    betas=(0.9, 0.999),    # 一阶和二阶矩的衰减率（β₁, β₂）
    eps=1e-8,              # 数值稳定性常数（默认 1e-8）
    weight_decay=0,        # L2 正则化系数（默认 0）
    amsgrad=False          # 是否使用 AMSGrad 变体（默认 False）
)
```

### **参数说明**

| 参数名            | 作用                                  |
| -------------- | ----------------------------------- |
| `params`       | 需要优化的参数（通常是 `model.parameters()`）。  |
| `lr`           | 初始学习率（典型值：`1e-3` 或 `5e-4`）。         |
| `betas`        | 动量衰减系数（`(β₁, β₂)`），控制一阶矩和二阶矩的指数衰减率。 |
| `eps`          | 防止除零的小常数（通常不修改）。                    |
| `weight_decay` | L2 正则化系数（等价于权重衰减）。                  |
| `amsgrad`      | 是否使用 AMSGrad 变体（解决 Adam 可能收敛不良的问题）。 |

***

## **3. 实际使用示例**

### **（1）基础用法**

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# 定义模型
model = nn.Linear(10, 1)

# 初始化 Adam 优化器
optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)

# 模拟训练步骤
for epoch in range(100):
    optimizer.zero_grad()          # 清空梯度
    inputs = torch.randn(32, 10)   # 随机输入
    targets = torch.randn(32, 1)   # 随机目标
    outputs = model(inputs)        # 前向传播
    loss = nn.MSELoss()(outputs, targets)
    loss.backward()                # 反向传播
    optimizer.step()              # 更新参数
```

### **（2）学习率调度**

```python
from torch.optim.lr_scheduler import StepLR

optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 每 30 轮学习率 ×0.1

for epoch in range(100):
    # ...（训练步骤）
    scheduler.step()  # 更新学习率
```

***

## **4. Adam 的特性与调参建议**

### **优点**

*   **自适应学习率**：每个参数有不同的学习率，适合稀疏梯度问题。
*   **动量加速**：结合了 Momentum 和 RMSProp 的优点。
*   **默认参数通用**：`lr=1e-3`、`betas=(0.9, 0.999)` 在大多数任务中表现良好。

### **调参技巧**

1.  **学习率（`lr`）**：
    *   初始值通常设为 `1e-3`，复杂任务可尝试 `5e-4` 或 `1e-4`。
    *   配合学习率调度器（如 `ReduceLROnPlateau`）动态调整。
2.  **权重衰减（`weight_decay`）**：
    *   防止过拟合，常用 `1e-4` 到 `1e-5`。
3.  **AMSGrad**：
    *   若训练不稳定，可尝试 `amsgrad=True`。

### **与其他优化器对比**

| 优化器         | 适用场景                  | 特点                      |
| ----------- | --------------------- | ----------------------- |
| **Adam**    | 大多数深度学习任务             | 自适应学习率 + 动量，收敛快，默认参数鲁棒。 |
| **SGD**     | 需要精细调参的任务（如训练 ResNet） | 依赖学习率调度和动量，可能收敛到更优解。    |
| **RMSProp** | 非平稳目标（如 RNN）          | 自适应学习率，无动量。             |

***

## **5. 数学推导补充**

### **偏差校正的作用**

Adam 的初始时刻 ( m\_0 = 0 )、( v\_0 = 0 )，导致早期更新偏向 0。通过偏差校正：

```math
\hat{m}\_t = \frac{m\_t}{1 - \beta\_1^t}, \quad \hat{v}\_t = \frac{v\_t}{1 - \beta\_2^t}
```

*   早期 ( t ) 较小时，分母 ( 1 - \beta^t ) 接近 0，放大 ( m\_t ) 和 ( v\_t )。
*   随着 ( t ) 增大，校正因子逐渐趋于 1。

### **AMSGrad 改进**

原始 Adam 的 ( v\_t ) 可能随时间下降，导致学习率上升。AMSGrad 通过保持历史最大 ( v\_t ) 避免此问题：

```math
v\_t^{\text{ams}} = \max(v\_{t-1}^{\text{ams}}, v\_t)
```

参数更新时分母使用 ( v\_t^{\text{ams}} )。

***

## **6. 总结**

*   **核心思想**：自适应学习率 + 动量，适合大多数深度学习任务。
*   **PyTorch 实现**：
    ```python
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    ```
*   **关键参数**：
    *   `lr`：控制更新步长。
    *   `betas`：控制动量衰减。
    *   `weight_decay`：L2 正则化。
*   **训练流程**：
    ```python
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```

Adam 因其鲁棒性和高效性，成为深度学习中的默认优化器之一。理解其原理后，可灵活调整参数以适应不同任务！
