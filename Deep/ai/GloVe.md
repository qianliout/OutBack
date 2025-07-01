## **GloVe（Global Vectors for Word Representation）详解**

GloVe 是一种基于全局统计信息的词向量表示方法，由斯坦福团队于2014年提出。它结合了**全局矩阵分解**（如LSA）和**局部上下文窗口**（如Word2Vec）的优点，通过统计共现矩阵和优化目标函数来生成词向量。

***

### **1. 核心思想**

GloVe 的目标是让词向量的点积尽可能接近词语共现次数的对数：

```math
\mathbf{w}_i^T \mathbf{w}_j + b_i + b_j \approx \log X_{ij}
```

其中：

*   ( `$\mathbf{w}\_i$` )：词语 ( i ) 的词向量
*   ( `$X\_{ij}$` )：词语 ( i ) 和 ( j ) 在语料库中的共现次数
*   ( `$b\_i, b\_j$` )：偏置项

***

### **2. 原理分步解释**

#### **步骤1：构建共现矩阵**

统计语料中所有词语对在固定窗口大小（如5-10词）内的共现次数。\
**示例数据**：句子 `"I love NLP and I love coding"`\
窗口大小=2，对称窗口：

|            | I | love | NLP | and | coding |
| ---------- | - | ---- | --- | --- | ------ |
| **I**      | 0 | 2    | 1   | 1   | 0      |
| **love**   | 2 | 0    | 1   | 1   | 1      |
| **NLP**    | 1 | 1    | 0   | 1   | 0      |
| **and**    | 1 | 1    | 1   | 0   | 1      |
| **coding** | 0 | 1    | 0   | 1   | 0      |

#### **步骤2：定义损失函数**

GloVe 的损失函数为加权最小二乘：

```math
J = \sum_{i,j=1}^V f(X_{ij}) \left( \mathbf{w}_i^T \mathbf{w}_j + b_i + b_j - \log X_{ij} \right)^2
```

*   ( `$f(X\_{ij})$` ) 是权重函数，减少高频词对损失的过度影响：
    ```math
    f(x) = \begin{cases} 
    (x/x_{\text{max}})^\alpha & \text{if } x < x_{\text{max}} \\
    1 & \text{otherwise}
    \end{cases}
    ```
    通常 ( `$x\_{\text{max}} = 100$` ), ( `$\alpha = 0.75$` )。

#### **步骤3：训练词向量**

通过梯度下降优化损失函数，得到词向量 ( `$\mathbf{w}\_i$` ) 和偏置项 ( `$b\_i$` )。

***

### **3. 实现步骤（Python示例）**

以下是用实际数据（英文维基百科片段）实现 GloVe 的简化代码：

#### **(1) 构建共现矩阵**

```python
from collections import defaultdict
import numpy as np

# 示例语料
corpus = [
    "I love NLP and I love coding".split(),
    "NLP is amazing".split(),
    "coding is fun".split()
]

# 统计共现次数
window_size = 2
cooccur = defaultdict(lambda: defaultdict(float))
vocab = set()

for sentence in corpus:
    for i, word in enumerate(sentence):
        vocab.add(word)
        start = max(0, i - window_size)
        end = min(len(sentence), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                cooccur[word][sentence[j]] += 1.0 / abs(i - j)  # 距离加权

vocab = list(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
X = np.zeros((len(vocab), len(vocab)))
for w1 in cooccur:
    for w2 in cooccur[w1]:
        X[word2idx[w1]][word2idx[w2]] = cooccur[w1][w2]

print("共现矩阵:\n", X)
```

**输出示例**：

    共现矩阵:
     [[0.  1.5 0.5 1.  0. ]
     [1.5 0.  1.  1.  1. ]
     [0.5 1.  0.  1.  0. ]
     [1.  1.  1.  0.  1. ]
     [0.  1.  0.  1.  0. ]]

#### **(2) 训练GloVe词向量**

```python
import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.w = nn.Embedding(vocab_size, embedding_dim)
        self.w_bias = nn.Embedding(vocab_size, 1)
        self.v = nn.Embedding(vocab_size, embedding_dim)
        self.v_bias = nn.Embedding(vocab_size, 1)

    def forward(self, i, j):
        return (self.w(i) * self.v(j)).sum(dim=1) + self.w_bias(i).squeeze() + self.v_bias(j).squeeze()

# 初始化模型
vocab_size = len(vocab)
embedding_dim = 10
model = GloVe(vocab_size, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 1000
for epoch in range(epochs):
    loss_val = 0
    for i in range(vocab_size):
        for j in range(vocab_size):
            if X[i][j] > 0:
                # 计算权重
                x = X[i][j]
                weight = (x / 100) ** 0.75 if x < 100 else 1
                # 前向传播
                pred = model(torch.tensor(i), torch.tensor(j))
                target = torch.log(torch.tensor(X[i][j]))
                loss = weight * (pred - target) ** 2
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_val}")

# 获取词向量
word_vectors = model.w.weight.data + model.v.weight.data
print("词向量形状:", word_vectors.shape)
```

***

### **4. 实际数据示例**

假设训练后得到词向量（简化版）：

| 词语     | 词向量（前3维）             |
| ------ | -------------------- |
| I      | \[0.12, -0.45, 0.67] |
| love   | \[0.89, -0.23, 0.11] |
| NLP    | \[0.34, 0.56, -0.78] |
| and    | \[-0.12, 0.33, 0.45] |
| coding | \[0.67, -0.89, 0.01] |

**语义关系验证**：

```python
# 计算 "love" 和 "coding" 的余弦相似度
cos_sim = torch.cosine_similarity(word_vectors[1:2], word_vectors[4:5], dim=1)
print(f"love 和 coding 的相似度: {cos_sim.item():.3f}")
```

输出可能是 `0.124`（因示例数据极小，实际需更大语料）。

***

### **5. 与传统方法对比**

| 特性       | Word2Vec (Skip-gram) | GloVe     |
| -------- | -------------------- | --------- |
| **训练目标** | 预测上下文词               | 拟合共现统计量   |
| **数据利用** | 局部窗口                 | 全局统计+局部窗口 |
| **计算效率** | 适合在线学习               | 需预计算共现矩阵  |
| **适用场景** | 大规模动态数据              | 静态语料库     |

***

### **6. 关键结论**

1.  GloVe 通过**全局共现统计**生成词向量，比 Word2Vec 更稳定。
2.  实现核心是**加权最小二乘优化**，平衡高频和低频词的影响。
3.  适合中等规模语料（如维基百科），但对超大数据集计算共现矩阵可能内存不足（需优化）。

## **GloVe损失函数公式详解**

GloVe的损失函数是理解其原理的核心。我们将通过**数学解释**、**物理意义**和**实例计算**三步，彻底解析这个公式：

***

### **1. 公式拆解**

原始公式：

```math
J = \sum_{i,j=1}^V f(X_{ij}) \left( \mathbf{w}_i^T \mathbf{w}_j + b_i + b_j - \log X_{ij} \right)^2
```

#### **(1) 各部分含义**

| 符号                    | 含义                                         |
| --------------------- | ------------------------------------------ |
| ( `$\mathbf{w}\_i$` ) | 词语 ( i ) 的词向量（目标向量）                        |
| ( `$\mathbf{w}\_j$` ) | 词语 ( j ) 的上下文向量（独立于 ( `$\mathbf{w}\_i$` )） |
| ( `$b\_i, b\_j$` )    | 词语 ( i ) 和 ( j ) 的偏置项（标量）                  |
| ( `$X\_{ij}$` )       | 词语 ( i ) 和 ( j ) 在语料中的共现次数                 |
| ( `$f(X\_{ij}) $`)    | 权重函数，用于平衡高频和低频词的影响                         |

#### **(2) 核心目标**

让词向量的点积 ( `$\mathbf{w}\_i^T \mathbf{w}*j $`) 逼近共现次数的对数 ( `$\log X*{ij}$` )，同时考虑偏置项。

***

### **2. 权重函数 ( f(X\_{ij}) ) 的作用**

#### **函数定义**：

```math
f(x) = \begin{cases} 
(x/x_{\text{max}})^\alpha & \text{if } x < x_{\text{max}} \\
1 & \text{otherwise}
\end{cases}
```

*   **典型值**：(`$ x\_{\text{max}} = 100 $`), ( `$\alpha = 0.75$` )
*   **设计意图**：
    *   避免高频词（如"the"）主导训练
    *   保护低频词（如专业术语）的信号

#### **示例计算**：

| 共现次数 ( X\_{ij} ) | ( f(X\_{ij}) ) 计算                        | 结果   |
| ---------------- | ---------------------------------------- | ---- |
| 5                | ( `$(5/100)^{0.75}$` )                   | 0.21 |
| 100              | ( 1 )（因为 (`$ x \geq x\_{\text{max}} $`)） | 1.0  |
| 200              | ( 1 )                                    | 1.0  |

***

### **3. 完整计算示例**

假设语料中：

*   词语对 ("cat", "dog") 共现 ( X\_{ij} = 8 ) 次
*   当前词向量和偏置：
    *   ( `$\mathbf{w}\_{cat} = [0.4, -0.2] $`)
    *   ( `$\mathbf{w}\_{dog} = [0.3, 0.6]$` )
    *   ( `$b\_{cat} = 0.1 ), ( b\_{dog} = -0.2 $`)

**步骤1：计算预测值**

```math
\mathbf{w}_{cat}^T \mathbf{w}_{dog} + b_{cat} + b_{dog} = (0.4 \times 0.3 + (-0.2) \times 0.6) + 0.1 + (-0.2) = -0.08
```

**步骤2：计算目标值**

```math
\log X_{ij} = \log 8 \approx 2.079
```

**步骤3：计算误差项**

```math
\text{误差} = -0.08 - 2.079 = -2.159
```

**步骤4：计算权重**\
假设 `$( x\_{\text{max}} = 10 ), ( \alpha = 0.75 )$`：

```math
f(8) = (8/10)^{0.75} \approx 0.85
```

**步骤5：计算单点损失**

```math
f(X_{ij}) \times \text{误差}^2 = 0.85 \times (-2.159)^2 \approx 3.96
```

***

### **4. 物理意义**

*   **词向量的点积**：衡量词语间的关联强度
*   **对数共现次数**：对真实统计量的平滑处理（因共现次数跨度大）
*   **偏置项**：捕获词语本身的全局频率偏差（如高频词本身可能与其他词共现多）

***

### **5. 为什么这样设计？**

| **设计选择**                                              | **原因**                         |
| ----------------------------------------------------- | ------------------------------ |
| **对数变换 ( `$\log X\_{ij} $`)**                         | 压缩共现次数的数值范围，使模型更稳定             |
| **分离的 ( `$\mathbf{w}\_i $`) 和 ( `$\mathbf{w}\_j $`)** | 更灵活地捕捉不对称关系（如"猫抓老鼠" vs "老鼠抓猫"） |
| **偏置项 (`$ b\_i, b\_j $`)**                            | 消除词语本身频率对共现的影响（如"的"高频但语义信息少）   |

***

### **6. 与Word2Vec的对比**

| **方面**   | GloVe           | Word2Vec (Skip-gram) |
| -------- | --------------- | -------------------- |
| **训练目标** | 直接拟合全局共现统计量     | 预测局部上下文窗口中的词         |
| **数据利用** | 显式使用整个语料的共现矩阵   | 通过滑动窗口隐式学习           |
| **计算效率** | 需预计算共现矩阵（内存消耗大） | 可在线学习（适合流数据）         |
| **典型结果** | 在中等规模语料上更稳定     | 超大规模语料表现更好           |

***

### **7. 实现时的注意事项**

1.  **共现矩阵的稀疏性**
    *   实际中99%的 ( X\_{ij} = 0 )，需用稀疏矩阵存储（如`scipy.sparse`）
2.  **权重函数的调参**
    *   (`$ x\_{\text{max}}$` ) 和 ( `$\alpha$` ) 影响低频词的学习效果
3.  **向量初始化**
    *   词向量通常用小的随机值初始化（如正态分布 `$( \mathcal{N}(0, 0.1) )$`)

***

### **8. 扩展：损失函数的另一种视角**

GloVe的损失函数可以看作是对以下关系的**加权最小二乘回归**：

```math
\mathbf{w}_i^T \mathbf{w}_j + b_i + b_j \approx \log X_{ij}
```

*   **回归目标**：用向量点积拟合对数共现次数
*   **权重**：(`$ f(X\_{ij}) $`) 确保不同频率的词对得到合理关注

***

### **总结**

GloVe的损失函数通过**全局共现统计**和**加权最小二乘优化**，将词语的语义关系编码到向量中。其核心在于：

1.  用点积逼近对数共现频率
2.  权重函数平衡高低频词
3.  偏置项消除频率偏差

这种设计使GloVe在小到中等规模语料上能高效捕捉线性语义关系（如"king - man + woman ≈ queen"）。
