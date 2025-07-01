### **PyTorch 中的 `nn.Embedding` 详解**

`nn.Embedding` 是 PyTorch 中用于**将离散的整数索引映射为连续向量表示**的类，常用于自然语言处理（NLP）中将单词、字符或类别标签转换为稠密向量（词嵌入）。以下是其作用、用法和参数的详细说明。

***

## **1. 核心作用**

*   **功能**：将整数索引（如单词ID）转换为固定维度的稠密向量。
*   **应用场景**：
    *   词嵌入（Word Embedding）。
    *   类别特征编码（如用户ID、商品ID）。
    *   任何需要将离散ID转换为向量的任务。

***

## **2. 类定义**

```python
torch.nn.Embedding(
    num_embeddings,      # 词典大小（最大索引值 +1）
    embedding_dim,       # 嵌入向量的维度
    padding_idx=None,    # 可选，填充符的索引（如<PAD>）
    max_norm=None,       # 可选，对嵌入向量做归一化的最大范数
    norm_type=2.0,       # 归一化的范数类型（默认L2）
    scale_grad_by_freq=False,  # 是否按词频缩放梯度
    sparse=False,        # 是否使用稀疏梯度更新（适用于大规模嵌入）
    _weight=None         # 可选，直接传入自定义的权重矩阵
)
```

***

## **3. 参数含义**

| 参数名                  | 类型     | 说明                                                      |
| -------------------- | ------ | ------------------------------------------------------- |
| `num_embeddings`     | int    | 嵌入字典的大小（即索引的最大值 +1）。例如，若有1000个单词，则设为1000。               |
| `embedding_dim`      | int    | 每个索引对应的嵌入向量的维度（如300维）。                                  |
| `padding_idx`        | int    | 指定某个索引为填充符（如0），其对应的嵌入向量会被强制设为0且不更新。                     |
| `max_norm`           | float  | 对嵌入向量做归一化时，允许的最大范数（超过则缩放）。                              |
| `norm_type`          | float  | 归一化的范数类型（如`2.0`表示L2范数）。                                 |
| `scale_grad_by_freq` | bool   | 若为`True`，梯度会根据词频缩放（低频词梯度更大）。                            |
| `sparse`             | bool   | 若为`True`，使用稀疏梯度更新（节省内存，但优化器需支持稀疏梯度）。                    |
| `_weight`            | Tensor | 可选，直接传入自定义的权重矩阵（形状需为`(num_embeddings, embedding_dim)`）。 |

***

## **4. 基本用法**

### **（1）初始化嵌入层**

```python
import torch
import torch.nn as nn

# 定义一个嵌入层：词典大小=10，嵌入维度=3
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# 查看权重矩阵（随机初始化）
print(embedding.weight)
# 输出：torch.Size([10, 3])
```

### **（2）将索引转换为嵌入向量**

```python
# 输入：整数索引（形状任意）
input_indices = torch.LongTensor([1, 4, 8])  # 3个索引

# 获取嵌入向量
output = embedding(input_indices)
print(output.shape)  # 输出：torch.Size([3, 3])
```

### **（3）处理批量数据**

```python
# 输入形状：[batch_size, seq_length]
batch_indices = torch.LongTensor([[1, 2, 3], [4, 5, 0]])

# 输出形状：[batch_size, seq_length, embedding_dim]
batch_output = embedding(batch_indices)
print(batch_output.shape)  # 输出：torch.Size([2, 3, 3])
```

***

## **5. 关键特性**

### **（1）填充符（`padding_idx`）**

```python
# 设置索引0为填充符（嵌入向量固定为0）
embedding = nn.Embedding(10, 3, padding_idx=0)

# 输入包含填充符
input_indices = torch.LongTensor([0, 2, 0])
output = embedding(input_indices)
print(output)
# 输出：tensor([[0., 0., 0.],  # 索引0的嵌入被强制为0
#               [x, x, x],  # 索引2的嵌入
#               [0., 0., 0.]], grad_fn=<EmbeddingBackward>)
```

### **（2）梯度缩放（`scale_grad_by_freq`）**

*   若设为`True`，梯度会根据词频自动调整，低频词的更新幅度更大。

### **（3）稀疏梯度（`sparse`）**

*   适用于大规模嵌入（如百万级词表），可节省内存，但需优化器支持（如`optim.SparseAdam`）。

***

## **6. 进阶用法**

### **（1）加载预训练词向量**

```python
# 假设已有预训练词向量（形状：[num_embeddings, embedding_dim]）
pretrained_weights = torch.randn(10, 3)

# 初始化嵌入层并载入权重
embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)  # freeze=True表示不更新
```

### **（2）自定义权重**

```python
custom_weights = torch.randn(10, 3)
embedding = nn.Embedding(10, 3, _weight=custom_weights)
```

### **（3）向量归一化（`max_norm`）**

```python
# 对嵌入向量做L2归一化（范数不超过1.0）
embedding = nn.Embedding(10, 3, max_norm=1.0)
```

***

## **7. 总结**

| **操作**  | **代码示例**                                          |
| ------- | ------------------------------------------------- |
| 初始化嵌入层  | `embedding = nn.Embedding(1000, 300)`             |
| 转换索引为向量 | `output = embedding(torch.LongTensor([1, 2, 3]))` |
| 处理批量数据  | `batch_output = embedding(batch_indices)`         |
| 设置填充符   | `padding_idx=0`                                   |
| 加载预训练向量 | `from_pretrained(weights, freeze=True)`           |
| 稀疏梯度更新  | `sparse=True`                                     |

### **何时使用 `nn.Embedding`？**

*   需要将离散ID（如单词、用户ID）映射为连续向量时。
*   需要微调或固定预训练词向量时。
*   处理类别特征时（替代One-Hot编码）。

通过灵活配置参数，`nn.Embedding` 可以高效地处理嵌入任务，是NLP和推荐系统中的基础组件。
