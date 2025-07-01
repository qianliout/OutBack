# 03_Transformer与注意力机制

## Transformer详解

# Transformer详解

## 概述

Transformer是2017年提出的革命性架构，完全摒弃循环结构，仅依赖**自注意力机制（Self-Attention）**和**前馈神经网络（Feed Forward Network, FFN）**实现序列建模。

## 1. Transformer的核心思想

### 自注意力机制（Self-Attention）

**动态权重分配**：每个词通过计算与其他所有词的关联权重（Attention Score），捕获全局依赖关系。

**数学表示**：
$$Attention(Q, K, V) = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$（Query）、$K$（Key）、$V$（Value）分别由输入线性变换得到
- $\sqrt{d_k}$ 用于缩放点积，防止梯度爆炸

### 多头注意力（Multi-Head Attention）

并行多组自注意力机制，捕获不同子空间的语义关系（如语法、语义、指代等）：

$$MultiHead(Q, K, V) = Concat(\text{head}_1, ..., \text{head}_h)W^O$$

每个 $\text{head}_i$ 独立计算注意力，最后拼接并线性变换。

### 位置编码（Positional Encoding）

通过正弦/余弦函数注入序列位置信息，弥补自注意力对词序不敏感的缺陷：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 残差连接与层归一化

每层输出为 $\text{LayerNorm}(x + \text{Sublayer}(x))$，缓解梯度消失并加速训练。

## 2. Self-Attention计算过程详解

### 步骤1：输入表示

假设输入序列包含 $n$ 个词，每个词表示为维度 $d_{\text{model}}$ 的向量，输入矩阵 $X \in \mathbb{R}^{n \times d_{\text{model}}}$：

$$X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

### 步骤2：计算Query, Key, Value矩阵

通过线性变换生成Query($Q$)、Key($K$)和Value($V$)矩阵：

$$\begin{aligned}
Q &= X W^Q, \quad W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k} \\
K &= X W^K, \quad W^K \in \mathbb{R}^{d_{\text{model}} \times d_k} \\
V &= X W^V, \quad W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
\end{aligned}$$

通常设 $d_k = d_v = d_{\text{model}} / h$（$h$ 为注意力头数）。

### 步骤3：计算注意力分数

通过 $Q$ 和 $K$ 的点积计算所有词对之间的相关性分数：

$$Scores = Q K^T \in \mathbb{R}^{n \times n}$$

为防止点积值过大导致梯度消失，除以 $\sqrt{d_k}$：

$$Scaled Scores = \frac{Q K^T}{\sqrt{d_k}}$$

### 步骤4：应用Softmax归一化

对每一行进行Softmax归一化，得到注意力权重矩阵 $A$：

$$A = Softmax\left(\frac{Q K^T}{\sqrt{d_k}}\right), \quad A \in \mathbb{R}^{n \times n}$$

权重 $A_{ij}$ 表示第 $i$ 个词对第 $j$ 个词的关注程度，每行权重和为1。

### 步骤5：加权求和Value矩阵

用注意力权重 $A$ 对 $V$ 加权求和，得到输出矩阵 $Z$：

$$Z = A V \in \mathbb{R}^{n \times d_v}$$

### 步骤6：多头注意力

将上述过程并行执行 $h$ 次，拼接所有头的输出并线性变换：

$$MultiHead(Q, K, V) = Concat(Z_1, Z_2, \dots, Z_h) W^O$$

其中 $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ 是输出投影矩阵。

## 3. 多头注意力的设计目的

### 核心设计目的

1. **捕捉多样化的依赖关系**
   - 头1可能学习**语法依赖**（如主语-动词一致性）
   - 头2可能学习**指代关系**（如代词与先行词）
   - 头3可能学习**局部词序**（如短语结构）

2. **提升模型表达能力**
   - 每个头将输入投影到不同的低维空间
   - 相当于多个"视角"观察数据

3. **增强鲁棒性**
   - 即使某些头失效，其他头仍能提供有效信息
   - 防止所有词向同一语义空间坍缩

### 数学表示

对于 $h$ 个头，每个头独立计算注意力：

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

最终拼接所有头的输出并线性变换：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

## 4. Transformer为何比RNN更高效？

| 对比维度 | RNN/LSTM | Transformer | Transformer的优势 |
|---------|----------|-------------|------------------|
| **长距离依赖** | 依赖循环逐步传递信息，梯度易消失/爆炸 | 自注意力直接建模任意距离关系 | 彻底解决长序列依赖问题 |
| **计算效率** | 必须按时间步顺序计算，无法并行 | 所有位置同时计算，硬件利用率高 | 训练速度提升5-10倍 |
| **内存消耗** | 隐藏状态需存储所有时间步中间结果 | 仅需存储注意力权重和激活值 | 更适应大规模数据 |
| **语义建模能力** | 局部上下文依赖较强，全局关系较弱 | 多头注意力捕获多层次语义 | 在BERT/GPT中实现SOTA性能 |

### 关键效率提升点

1. **并行化计算**
   - RNN的 $h_t$ 依赖 $h_{t-1}$，必须串行计算
   - Transformer的注意力矩阵可一次性计算所有位置的关联

2. **信息传递路径**
   - RNN中两个词的交互需经过 $O(n)$ 步
   - Transformer中任意两词直接交互（路径长度 $O(1)$）

3. **梯度传播**
   - RNN的梯度通过时间（BPTT）易衰减
   - Transformer的残差连接保持梯度稳定

## 5. Transformer的典型应用

- **编码器架构（如BERT）**：文本分类、命名实体识别（NER）
- **解码器架构（如GPT）**：文本生成、代码补全
- **完整Seq2Seq（如T5、BART）**：机器翻译、文本摘要

## 6. PyTorch实现示例

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(attention_output)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + 层归一化
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## 7. 计算复杂度分析

### 时间复杂度
- **Self-Attention**: $O(n^2 \cdot d_{\text{model}})$
- **多头注意力**: $O(n^2 \cdot d_{\text{model}} + n \cdot h \cdot d_k \cdot d_v)$

### 空间复杂度
- **注意力权重矩阵**: $O(n^2)$
- **总空间**: $O(n^2 \cdot d_{\text{model}})$

## 8. 优化技巧

### 1. 稀疏注意力
- **局部注意力**: 只关注局部窗口内的token
- **稀疏变换器**: 减少注意力计算量

### 2. 线性注意力
- 使用核函数近似，将复杂度从 $O(n^2)$ 降低到 $O(n)$

### 3. 相对位置编码
- 替代绝对位置编码，更好地处理长序列

## 9. 面试回答技巧

- **核心思想总结**：
  > "Transformer通过自注意力机制实现全局上下文建模，用位置编码保留序列信息，最终以并行计算和残差连接解决了RNN的效率和长依赖瓶颈。"

- **举例说明高效性**：
  > "在翻译一篇500词的文档时，RNN需逐步传递信息导致首尾语义断裂，而Transformer的注意力机制能直接关联开头和结尾的关键词。"

- **引申到现代模型**：
  > "这一设计催生了BERT和GPT等模型，成为NLP领域的基石架构。" 

