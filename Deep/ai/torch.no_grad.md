## **`torch.no_grad()` 的作用与常用用法详解**

***

### **1. 核心作用**

**`torch.no_grad()`** 是 PyTorch 的一个上下文管理器（Context Manager），用于**禁用梯度计算**，主要功能包括：

1.  **减少内存消耗**：不跟踪计算图中的梯度，节省显存。
2.  **加速计算**：避免不必要的梯度计算和反向传播。
3.  **防止误修改**：保护模型参数不被意外更新（如评估时）。

**数学原理**：\
在 `no_grad()` 上下文中，所有张量操作的 `requires_grad=False`，PyTorch 不会构建计算图（Computation Graph）。

***

### **2. 使用场景**

#### **(1) 模型评估（Inference）**

在测试或验证阶段，无需计算梯度：

```python
model.eval()  # 设置模型为评估模式（关闭Dropout/BatchNorm等）
with torch.no_grad():
    outputs = model(inputs)  # 前向传播，不记录梯度
    predictions = outputs.argmax(dim=1)
```

#### **(2) 冻结参数时计算中间结果**

例如在特征提取或迁移学习中：

```python
for param in model.parameters():
    param.requires_grad = False  # 冻结所有参数

with torch.no_grad():
    features = model.conv_layers(inputs)  # 只提取特征，不更新梯度
```

#### **(3) 显式避免梯度计算**

当需要临时禁用梯度时（如手动更新参数）：

```python
x = torch.tensor([1.0], requires_grad=True)
with torch.no_grad():
    y = x * 2  # y.requires_grad = False，即使x需要梯度
```

***

### **3. 与相关函数的对比**

| 方法                              | 作用                          | 是否影响模型参数 |
| ------------------------------- | --------------------------- | -------- |
| `torch.no_grad()`               | 禁用梯度计算，不保存计算图               | 不影响      |
| `model.eval()`                  | 关闭Dropout/BatchNorm的推理模式    | 不影响      |
| `torch.set_grad_enabled(False)` | 全局禁用梯度（功能同`no_grad()`，但更灵活） | 不影响      |

***

### **4. 常见用法示例**

#### **(1) 评估模型准确率**

```python
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')
```

#### **(2) 提取特征（不更新参数）**

```python
features_list = []
model.eval()
with torch.no_grad():
    for data in dataloader:
        features = model.feature_extractor(data)  # 假设feature_extractor是子模块
        features_list.append(features)
```

#### **(3) 手动更新参数（跳过梯度跟踪）**

```python
with torch.no_grad():
    for param in model.parameters():
        param += 0.01 * torch.randn_like(param)  # 直接修改参数，不触发梯度
```

***

### **5. 注意事项**

1.  **与`model.eval()`的区别**：
    *   `model.eval()` 改变模型行为（如关闭Dropout），但不影响梯度计算。
    *   `no_grad()` 禁用梯度，但不会自动关闭Dropout。\
        **通常两者需同时使用**：
    ```python
    model.eval()
    with torch.no_grad():
        # 评估代码
    ```

2.  **嵌套使用**：
    ```python
    with torch.no_grad():
        y = x * 2
        with torch.no_grad():  # 内层no_grad()冗余但合法
            z = y + 1
    ```

3.  **错误示例**：
    ```python
    x = torch.tensor([1.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    y.backward()  # 报错！y无梯度计算历史
    ```

***

### **6. 性能对比（显存/速度）**

| 操作          | 显存占用 | 速度 |
| ----------- | ---- | -- |
| 启用梯度（默认）    | 高    | 慢  |
| `no_grad()` | 低    | 快  |

**实测建议**：在批量推理时，使用`no_grad()`可显著减少显存占用（尤其对大模型）。

***

## **总结**

*   **何时用**：评估、特征提取、手动参数更新等无需梯度的场景。
*   **怎么用**：
    ```python
    with torch.no_grad():
        # 禁用梯度的代码块
    ```
*   **为什么用**：节省资源、加速计算、防止误操作。

**扩展阅读**：

*   官方文档：[`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
*   高级用法：[`torch.inference_mode()`](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)（PyTorch 1.9+，更严格的优化模式）

