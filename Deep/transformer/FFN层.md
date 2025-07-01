# FFN 层 (Feed-Forward Network)

## 1. 实现原理

FFN 层，全称为**位置相关前馈网络（Position-wise Feed-Forward Network）**，是 Transformer 的 Encoder 和 Decoder 中，继自注意力子层之后的第二个核心子层。它的作用是对自注意力层聚合了全局上下文信息后的输出，进行一次**非线性的、深度的加工和提炼**。

**架构：**

FFN 的结构非常简单，它就是一个小型的、两层的全连接神经网络（也称为多层感知机，MLP）。对于输入序列中的**每一个位置（position）**的向量，它都**独立地、相同地**应用这个两层网络。

其计算过程如下：

1.  **第一个线性变换:** 将输入向量 `x`（维度为 `d_model`）通过一个线性层（权重为 `W_1`，偏置为 `b_1`），将其维度从 `d_model` 扩展到一个更大的中间维度 `d_ff`。

    `output_1 = Linear_1(x) = x * W_1 + b_1`

2.  **非线性激活:** 对第一个线性层的输出应用一个非线性激活函数，最常用的是 **ReLU (Rectified Linear Unit)**，但也有研究使用 GELU (Gaussian Error Linear Unit) 或 Swish 等。

    `activated_output = Activation(output_1)`

3.  **第二个线性变换:** 将激活后的中间层向量，通过第二个线性层（权重为 `W_2`，偏置为 `b_2`），将其维度从 `d_ff` 重新映射回原始的 `d_model`。

    `FFN(x) = Linear_2(activated_output) = activated_output * W_2 + b_2`

**关键特性：**

*   **位置相关 (Position-wise):** 这个 FFN 网络是以“位置”为单位独立应用的。也就是说，序列中第 `i` 个位置的 token 只会通过 FFN 得到它自己的新表示，与其他位置的 token 在 FFN 内部没有直接交互。所有位置共享**同一套** `W_1, b_1, W_2, b_2` 权重。
*   **维度扩展:** 在原始的 Transformer 论文中，中间层的维度 `d_ff` 通常被设置为 `d_model` 的 **4 倍**（例如，`d_model=512`, `d_ff=2048`）。这种“先扩大再缩小”的沙漏型结构，被认为可以帮助模型学习到更丰富的特征组合，并增加模型的非线性表达能力。

在整个 Transformer Block 中，FFN 子层的完整流程是：
`Input -> LayerNorm -> FFN -> Dropout -> Residual Connection`

---

## 2. 所解决的问题

如果说自注意力层的作用是**“聚合信息”**（从全局收集相关的上下文），那么 FFN 层的作用就是**“处理和提炼信息”**。

1.  **增加模型的非线性能力:** 自注意力机制本身，如果移除 Softmax，本质上是对 Value 向量的加权求和，是一种线性变换。虽然 Softmax 引入了非线性，但 FFN 提供的强大的、参数化的非线性变换能力，是模型能够拟合复杂数据分布的关键。没有 FFN，深度 Transformer 将难以训练和工作。

2.  **转换和丰富特征表示:** FFN 层可以被看作是一个特征提取器。它将注意力层输出的、混合了各种信息的向量，映射到一个更高维的空间中，进行特征的交叉和组合，然后再压缩回原始维度，从而得到一个更具信息量、更适合下一层处理的表示。

3.  **提供可学习的参数:** FFN 层包含了大量的可学习参数（`W_1, W_2`），是 Transformer 模型参数的主要来源之一。这为模型提供了足够的容量（capacity）来记忆和学习语言中的复杂模式和世界知识。

---

## 3. 核心代码

在 PyTorch 中，FFN 层可以非常容易地通过 `torch.nn.Sequential` 容器和 `torch.nn.Linear` 模块来实现。

```python
import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): 输入和输出的维度.
            d_ff (int): 中间层的维度 (通常是 d_model * 4).
            dropout (float): Dropout 的概率.
        """
        super(PositionWiseFeedForward, self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), # 第一个线性层
            nn.ReLU(),                # 激活函数
            nn.Dropout(dropout),      # Dropout 层
            nn.Linear(d_ff, d_model)  # 第二个线性层
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return self.ffn(x)

# 在 EncoderLayer 中的使用
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 注意力子层
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN 子层
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

---

## 4. 实际工程中的应用

FFN 层是所有 Transformer 架构的**标准和核心组件**。

*   **所有主流模型:** BERT, GPT, T5, LLaMA 等无一例外地在其每个 Transformer Block 中都包含了 FFN 层。
*   **模型参数的主要来源:** 在一个典型的 Transformer 模型中，FFN 层的参数量通常占到总参数量的 **2/3** 左右。因此，它也是模型扩展（scaling up）时的一个主要对象。
*   **研究和优化的热点:**
    *   **激活函数:** 研究者们探索了使用 GELU, Swish/SiLU 等替代 ReLU，并发现在大型模型上通常能带来性能提升。现代 LLM 大多使用这些改进的激活函数。
    *   **MoE (Mixture of Experts):** 为了在不显著增加计算量的情况下，进一步扩大模型参数量，研究者们提出了 MoE 架构。其核心思想是将一个大的 FFN 层替换为多个小型的“专家”FFN 层，并通过一个可学习的“路由器”（gating network）来为每个 token 动态地选择激活少数几个专家。像 Mixtral, Switch Transformer 等模型就采用了这种结构，实现了参数量的高效扩展。

总而言之，FFN 层是 Transformer 中与自注意力层同等重要的核心模块，它为模型提供了强大的非线性处理能力和表示容量，是模型能够学习和理解复杂数据模式的关键所在。
