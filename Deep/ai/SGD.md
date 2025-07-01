# PyTorch中的SGD优化器详解

## SGD基本概念

SGD（Stochastic Gradient Descent，随机梯度下降）是深度学习中最基础的优化算法，PyTorch中的`torch.optim.SGD`实现了这一算法及其多种变体。

## 基本使用方法

```python
optimizer = torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

## 参数详解

## 1. 必需参数

| 参数名      | 类型       | 说明                               |
| -------- | -------- | -------------------------------- |
| `params` | iterable | 需要优化的参数（通常是`model.parameters()`） |
| `lr`     | float    | **学习率**（learning rate），控制参数更新步长  |

## 2. 可选参数

| 参数名            | 默认值   | 说明                                  |
| -------------- | ----- | ----------------------------------- |
| `momentum`     | 0     | **动量因子**（0≤m<1），加速相关方向的梯度下降         |
| `dampening`    | 0     | 动量抑制因子（0≤dampening<1），与momentum配合使用 |
| `weight_decay` | 0     | **L2惩罚系数**（权重衰减）                    |
| `nesterov`     | False | 是否使用**Nesterov动量**                  |

## 参数详细解释

## 1. 学习率（lr）

*   **作用**：控制每次参数更新的步长
*   **典型值**：0.01、0.001、0.1（不同任务差异大）
*   **调整建议**：
    *   太大：可能导致震荡或不收敛
    *   太小：收敛速度慢
    *   常用学习率衰减策略：StepLR、ReduceLROnPlateau

## 2. 动量（momentum）

*   **原理**：
    ```math
    v_{t} = \mu \cdot v_{t-1} + g_{t}
    ```
    ```math
    \theta_{t+1} = \theta_{t} - \eta \cdot v_{t}
    ```

*   **变量说明**：

    *   `v_t`：当前时刻的动量向量（累积梯度）
    *   `v_{t-1}`：上一时刻的动量向量
    *   `μ`（mu）：动量系数（通常取值0.9）
    *   `g_t`：当前时刻的梯度（∇J(θ\_t)）

*   **物理意义**：

    *   类似于物体运动中的速度概念
    *   当前速度 = 衰减后的历史速度 + 当前加速度（梯度）

*   **变量说明**：

    *   `θ_t`：当前时刻的参数值
    *   `η`（eta）：学习率
    *   `v_t`：当前动量

*   **物理意义**：

    *   参数移动 = 当前位置 - 学习率 × 当前速度



*   **作用**：
    *   加速收敛
    *   减少震荡
    *
*   **典型值**：0.9（CNN常用）、0.99（RNN有时用更高值）

## 3. 阻尼（dampening）

*   **作用**：抑制动量项
    ```math
    v_{t} = \mu \cdot v_{t-1} + (1 - \text{dampening}) \cdot g_{t}
    ```
*   **典型设置**：
    *   通常保持0
    *   当momentum接近1时可设为小值（如0.1）

## 4. 权重衰减（weight\_decay）

*   **原理**：L2正则化项
    ```math
    \theta_{t+1} = \theta_t - \eta \cdot (\nabla J(\theta_t) + \lambda \theta_t)
    ```
*   **作用**：防止过拟合
*   **典型值**：1e-4、5e-4

## 5. Nesterov动量（nesterov）

*   **原理**：改进版动量
    ```math
    v_{t} = \mu \cdot v_{t-1} + g_{t}(\theta_{t} - \mu \cdot v_{t-1})
    ```
*   **特点**：
    *   在动量方向上"前瞻"一步
    *   通常比普通momentum收敛更快
*   **使用条件**：必须`momentum>0`才能启用

## 不同配置的SGD变体

1.  **普通SGD**：
    ```python
    SGD(params, lr=0.1)
    ```

2.  **带动量的SGD**：
    ```python
    SGD(params, lr=0.1, momentum=0.9)
    ```

3.  **Nesterov动量SGD**：
    ```python
    SGD(params, lr=0.1, momentum=0.9, nesterov=True)
    ```

4.  **权重衰减SGD**：
    ```python
    SGD(params, lr=0.1, weight_decay=1e-4)
    ```

## 实际应用示例

## CNN训练典型配置

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)
```

## 不同参数组设置

```python
# 对卷积层和全连接层使用不同学习率
optimizer = torch.optim.SGD(
    [
        {'params': model.conv_params, 'lr': 0.01},
        {'params': model.fc_params, 'lr': 0.001}
    ],
    momentum=0.9,
    weight_decay=1e-4
)
```

## 参数选择建议

1.  **学习率**：
    *   先用较大值（如0.1）测试收敛性
    *   配合学习率调度器调整

2.  **动量**：
    *   视觉任务：0.9
    *   NLP任务：0.99（当使用较大batch size时）

3.  **权重衰减**：
    *   CNN：1e-4到5e-4
    *   Transformer：0.01（常配合AdamW使用）

4.  **Nesterov**：
    *   当使用动量时推荐启用
    *   对小批量数据效果更明显

## 与其他优化器的对比

1.  **相比Adam**：
    *   SGD+动量通常需要更多调参
    *   最终精度可能更高（特别是CV任务）
    *   更适合预训练模型微调

2.  **相比RMSprop**：
    *   SGD对学习率更敏感
    *   RMSprop自动调整各参数学习率

需要了解如何在训练过程中动态调整这些参数，或者想了解特定任务（如目标检测、GAN等）的最佳SGD配置吗？
