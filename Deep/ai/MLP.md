# &#x20;实现

    class MLP(d2l.Classifier):
        def __init__(self, num_outputs, num_hiddens, lr):
            super().__init__()
            self.save_hyperparameters()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(num_hiddens),
                nn.ReLU(),
                nn.LazyLinear(num_outputs),
            )


    if __name__ == "__main__":
        model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
        data = d2l.FashionMNIST(batch_size=256)
        trainer = d2l.Trainer(max_epochs=10)
        trainer.fit(model, data)

# &#x20;各层的权重是怎么初始化的

在PyTorch的`nn.LazyLinear`层中，权重的初始化是自动完成的。具体来说：

1.  **LazyLinear层的初始化特点**：

*   使用`nn.LazyLinear`时，PyTorch会延迟初始化权重矩阵
*   在第一次前向传播时，根据输入数据的形状自动确定输入维度
*   然后按照PyTorch默认的初始化策略初始化权重

1.  **默认初始化规则**：

*   权重矩阵：采用Kaiming均匀初始化（针对ReLU激活函数优化）
*   偏置项：初始化为0

1.  **等效于手动初始化**：

```python
# 相当于这样的初始化过程
weight = nn.init.kaiming_uniform_(torch.empty(out_features, in_features))
bias = nn.init.zeros_(torch.empty(out_features))
```

1.  **初始化时机**：

*   发生在第一次调用`forward()`时
*   输入形状为`(batch_size, in_features)`时自动确定`in_features`
*   之后权重矩阵形状固定为`(out_features, in_features)`

1.  **验证方法**：
    可以在第一次前向传播后检查权重：

```python
print(model.net[1].weight.shape)  # 第一个全连接层权重
print(model.net[3].weight.shape)  # 输出层权重
```

这种设计既保证了灵活性（无需手动指定输入维度），又遵循了深度学习的最佳初始化实践。
