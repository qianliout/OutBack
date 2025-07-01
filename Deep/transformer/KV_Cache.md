# KV Cache

## 1. 实现原理

KV Cache 是一种在 Transformer Decoder 进行自回归生成时，用来加速计算的关键优化技术。它的全称是 **Key-Value Cache**。

**背景：自回归生成的计算瓶颈**

在自回归生成过程中，每生成一个新的 token，模型都需要将**所有**已经生成的 token（包括新的这一个）作为输入，重新进行一次完整的自注意力计算。例如：
*   生成第 1 个 token 时，输入是 `[t_0]`
*   生成第 2 个 token 时，输入是 `[t_0, t_1]`
*   生成第 3 个 token 时，输入是 `[t_0, t_1, t_2]`
*   ... ...
*   生成第 `N` 个 token 时，输入是 `[t_0, ..., t_{N-1}]`

在计算第 `N` 个 token 的注意力时，我们需要用它的 Query 向量 `q_N` 与**所有**历史 token（包括自己）的 Key 向量 `[k_0, k_1, ..., k_N]` 进行点积。可以发现，其中 `[k_0, ..., k_{N-1}]` 这部分的 Key 向量，以及它们对应的 Value 向量 `[v_0, ..., v_{N-1}]`，在之前的步骤中其实**已经计算过**了。随着生成序列变长，这种重复计算会造成巨大的浪费，使得生成速度越来越慢。

**KV Cache 的核心思想：**

**缓存（Cache）并重用（Reuse）** 那些不会改变的 Key 和 Value 向量。

具体实现流程如下：

1.  **在第一次（或预填充）计算时：**
    *   模型接收初始输入（例如，Prompt）。
    *   在 Decoder 的每一个自注意力层中，正常计算所有输入 token 对应的 Key 和 Value 向量。
    *   在计算完成后，**将这些 Key 和 Value 向量（即 K 和 V 矩阵）存储在一个“缓存”中**。这个缓存通常是一个与层数、批次大小、序列长度、头数和维度相关的张量。

2.  **在后续的生成步骤中（例如，生成第 `N+1` 个 token）：**
    *   模型只需要将**新生成的那个 token**（即第 `N` 个 token）作为输入进行前向传播。
    *   在自注意力层中，只计算这个新 token 对应的 Key 和 Value 向量（`k_N` 和 `v_N`）。
    *   将新的 `k_N` 和 `v_N` **追加（append）** 到之前缓存的 K 和 V 矩阵的末尾，形成更新后的 `K_cache = [k_0, ..., k_N]` 和 `V_cache = [v_0, ..., v_N]`。
    *   然后，用新 token 的 Query 向量 `q_N` 与**完整的 `K_cache`** 进行注意力计算，并用得到的注意力权重与**完整的 `V_cache`** 进行加权求和。

通过这种方式，每次生成新的 token 时，计算量从与整个已生成序列长度 `N` 相关，**降低到了只与单个新 token 相关**。这极大地减少了冗余计算，显著提升了长序列的生成速度。

---

## 2. 所解决的问题

KV Cache 主要解决了**自回归生成过程中的推理效率低下**的问题。

*   **避免重复计算:** 它是解决自回归生成中冗余计算问题的最直接、最有效的方法。
*   **大幅提升推理速度:** 特别是在生成长文本（如长篇文章、代码、对话）时，KV Cache 带来的速度提升是数量级的。没有 KV Cache，现代大型语言模型（LLM）的在线交互式应用几乎是不可能实现的。
*   **降低显存占用（在某些方面）:** 虽然 Cache 本身需要占用显存，但由于每次迭代的输入序列长度从 `N` 变成了 `1`，这减少了在前向传播过程中因中间激活值而产生的动态显存占用。

---

## 3. 核心代码

在实际应用中，KV Cache 的支持通常已经内置在主流的深度学习框架或模型库（如 Hugging Face Transformers）中。在使用 `model.generate()` 方法时，`use_cache` 参数默认就是开启的。

下面是一个简化的伪代码，展示了在模型内部如何处理 KV Cache。

```python
# 伪代码，示意在多头注意力模块的 forward 方法中如何处理 KV Cache

def forward(self, x, past_key_value=None): # x 是当前输入, e.g., shape [batch, 1, embed_dim]
    # ... 计算 Q, K, V
    # query, key, value 的 shape 都是 [batch, num_heads, 1, head_dim]
    query = self.q_proj(x).view(...)
    key = self.k_proj(x).view(...)
    value = self.v_proj(x).view(...)

    if past_key_value is not None:
        # 如果有缓存，则取出过去的 K, V
        past_key = past_key_value[0]
        past_value = past_key_value[1]
        
        # 将新的 K, V 与过去的 K, V 拼接
        # key shape from [b, h, 1, d] to [b, h, seq_len, d]
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)

    # 将当前的 K, V 保存起来，以备下次使用
    present_key_value = (key, value)

    # ... 使用完整的 key 和 value 进行注意力计算 ...
    # attn_output = F.scaled_dot_product_attention(query, key, value, ...)

    # 返回输出和更新后的 KV Cache
    return attn_output, present_key_value

# 在整个模型的生成循环中
# past_kv_cache = None
# for i in range(max_len):
#     output, past_kv_cache = model_layer(input, past_key_value=past_kv_cache)
#     input = ... # 根据 output 确定下一个输入 token
```

---

## 4. 实际工程中的应用

KV Cache 是所有基于 Transformer 的自回归生成模型在进行**高效推理**时的**标准配置和核心优化**。

*   **大型语言模型 (LLM):** 所有提供在线聊天、文本生成服务的 LLM（如 ChatGPT, Claude）都在其推理引擎中深度集成了 KV Cache 技术。
*   **开源框架:** Hugging Face 的 `transformers` 库，NVIDIA 的 FasterTransformer 和 TensorRT-LLM，以及其他各种推理优化框架，都将 KV Cache 作为其核心优化之一。
*   **硬件加速:** 现代 GPU 的架构也越来越适合于 KV Cache 这类内存密集型操作的加速。

**KV Cache 的挑战:**

*   **显存占用:** KV Cache 本身会占用大量的显存，其大小与批次大小、序列长度、模型层数和头数成正比。对于非常长的序列，KV Cache 可能成为显存瓶颈。为了解决这个问题，研究者们提出了**多查询注意力 (Multi-Query Attention, MQA)** 和 **分组查询注意力 (Grouped-Query Attention, GQA)** 等技术，通过让多个查询头共享同一份 Key 和 Value，来大幅减少 Cache 的大小。

总而言之，KV Cache 是将 Transformer 从一个强大的理论模型转变为能够被广泛、高效地应用于实际生产环境的关键技术之一。
