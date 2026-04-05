# 大语言模型推理过程详解：从Prompt输入到结果返回

## 1. 问题背景与核心框架

### 面试问题

> "描述一下一个请求 prompt 经过 LLM 直到返回结果，这中间的推理过程，越详细越好。"

### 核心框架

一个 prompt 从输入到输出，大体会经历 **6 个阶段**：

1. 请求封装
2. Tokenization
3. 推理调度
4. Prefill
5. Decode
6. 结果反解码返回

**核心本质**：模型先并行"读懂"整段输入，建立上下文状态和 KV cache，然后再进入自回归生成循环，每次只预测下一个 token。

> 这种"自回归 + 不做本次梯度更新"的推理方式，正是 GPT 类语言模型的基本范式；而 Transformer 则提供了它内部 attention 和前馈网络的计算骨架。

## 2. 详细阶段解析

### 第一阶段：请求封装 - 用户输入 ≠ 模型输入

- **关键点**：我们在聊天框里看到的是自然语言，但模型真正接收到的，通常不是这段原始文本本身。
- **处理流程**：
    1. 服务层先把 system、user、assistant 等多轮消息按固定模板组织起来
    2. 补上一些特殊标记（如开始/结束标记）
    3. 将组织好的文本送入tokenizer进行分词
- **重要性**：后面所有推理，都是建立在 token 序列上的。无论输入是中文、英文还是代码，第一步都需转换成 token IDs。

### 第二阶段：Tokenization - 文本转离散标记

- 使用 tokenizer（如 OpenAI 的 tiktoken）将文本切割成 token 序列
- tokenizer 通常基于 BPE (Byte Pair Encoding) 算法
- **输出**：离散的 token IDs，而非原始文本
- **工程实践**：
    - 在现代推理服务中，tokenizer 通常与 serving 引擎绑定
    - 用户通常发送原始字符串，推理服务负责编码成 token IDs
    - 某些架构下，tokenization 会提前在客户端或独立预处理层完成
    - 高级系统如 vLLM 支持 text prompt 和 pre-tokenized prompt 两种模式

### 第三阶段：推理调度 - 请求排队与优化

- **关键概念**：请求不会立刻进模型，而是先进入推理服务和调度层
- **推理框架作用**（如 TGI、vLLM）：
    - 请求排队
    - 动态 batching (continuous batching)
    - 缓存管理
    - 流式返回
- **核心技术**：
    - Continuous batching：提高 GPU 利用率、降低延迟，允许请求在每一步动态加入和退出批次
    - Flash Attention：优化 attention 计算
    - Paged Attention：改进缓存管理
    - Token streaming：支持流式响应
- **系统链路**：
    
    ```
    用户输入 → prompt 模板展开 → tokenization → 请求调度 / batching → 送入模型
    ```
    

### 第四阶段：向量嵌入 - 离散到连续

- **Embedding lookup**：将 token IDs 映射成高维向量
    - 每个 token 会查一张巨大的 embedding 表，得到自己的向量表示
    - 此时模型才真正进入连续空间的数值计算
- **位置编码**：
    - 早期 Transformer 使用固定位置编码
    - 现代大模型多使用 RoPE (Rotary Position Embedding)
    - RoPE 将位置信息融入 attention 计算中，保留相对位置信息

### 第五阶段：Transformer 计算 - 模型核心

- **基础结构**：一个典型的 decoder-only LLM，每一层包含：
    1. Self-Attention
    2. FFN/MLP（前馈网络）
    3. 中间配合残差连接和归一化

#### Self-Attention 机制

- **目的**：当前位置的 token，要去看上下文里哪些 token 最相关
- **处理流程**：
    1. 将当前隐藏状态投影成 Query、Key、Value 三组向量
    2. 通过 Query 和所有 Key 的相似度计算注意力权重
    3. 对 Value 做加权求和
- **Causal Mask**：当前位置只能看见自己和前面的 token，不能偷看未来
    - 这一机制决定了模型天然是自回归生成的
- **通俗理解**（图书馆检索比喻）：
    - Q (Query) = 你现在脑子里有一个问题
    - K (Key) = 书架上每本书卡片上的主题标签
    - V (Value) = 书里真正的内容

#### FFN (前馈网络) 作用

- **功能**：对每个 token 的表示单独做非线性变换，把特征进一步提纯和增强
- **与 Attention 的区别**：
    - Attention 负责"从上下文搬运信息"
    - FFN 负责"对当前位置做进一步加工"
- **特点**：不会跨位置交互，是 position-wise 的操作

### 第六阶段：Prefill - 整体理解输入

- **定义**：先把整段 prompt 一次性跑完整个前向过程
- **工作内容**：
    - 为输入中的所有 token 计算各层隐藏状态
    - 生成后面 decode 要用到的 KV cache
- **特点**：
    - 高度并行：整段输入已完整给定，GPU 能把很多矩阵操作一起做完
    - Compute-bound：更偏向计算密集型
- **通俗比喻**：像考试时先读题，把题目读到脑子里，填充好上下文，然后再开始做答

### 第七阶段：KV Cache - 高效生成的关键

- **核心价值**：避免每生成一个新 token 时，将整个历史上下文从头再算一遍
- **工作原理**：
    - 历史 token 在每层 attention 中算出的 K 和 V 被缓存起来
    - 新 token 到来时，只需为这个新 token 计算新的 Query、Key、Value
    - 用新的 Query 去和历史缓存里的 Key 做匹配
- **为什么只缓存 K 和 V，不缓存 Q**：
    - 缓存决策依据："后面还会不会再被用到"
    - K、V 会在后面每一步继续被反复用到
    - Q "只在当前这一步有用一次"
- **通俗比喻**：
    - 没有 KV cache：像每次都重读整篇文章
    - 有 KV cache：像前文已经做好笔记，现在只补最后一句

### 第八阶段：Decode - 逐 token 生成答案

- **开始条件**：prefill 完成后，模型已经"读懂"了整段输入
- **生成流程**：
    1. 取最后一个位置的隐藏状态
    2. 通过输出层映射成整个词表上的 logits（"下一个 token 的打分"）
    3. 通过 softmax 和解码策略，决定下一个 token 输出什么
- **解码策略**：
    - Greedy：选择概率最大的 token
    - Sampling：根据概率分布随机采样
    - Top-k：只从概率最高的 k 个 token 中采样
    - Top-p (nucleus sampling)：从累积概率超过 p 的最小 token 集合中采样
- **循环过程**：
    
    ```
    1. 把刚生成的 token 接到上下文后面
    2. 复用 KV cache
    3. 只为这个新 token 跑一遍前向计算
    4. 再得到新的 logits
    5. 再生成下一个 token
    ```
    
- **现象解释**：为什么回答总是一个 token 一个 token 流式地吐出来，而不是整段瞬间出现

### 第九阶段：结果反解码与返回

- 将生成的 token 序列通过 tokenizer 反解码成自然语言文本
- 通过推理服务框架流式返回给客户端
- 处理特殊标记（如结束标记）以决定何时终止生成

## 3. 性能现象与优化

### 为什么"第一个字慢，后面快"？

- **Prefill 阶段**：
    
    - Compute-bound：可以并行处理多个 token
    - 吃满 GPU 算力，充分利用硬件
    - 处理整段 prompt 需要大量计算
- **Decode 阶段**：
    
    - Memory-bound：强依赖历史 KV cache，频繁访问显存
    - 严格的顺序依赖：必须等前一个 token 生成完成后才能计算下一个
    - 每步只计算一个 token，无法充分利用 GPU 并行能力

### 工程优化技术

- **FlashAttention**：通过 IO-aware 的 attention 计算方式，减少显存读写
- **Continuous batching**：动态调整批次，减少 GPU 空转
- **Chunked prefill**：改进长上下文处理
- **Paged Attention**：优化缓存管理效率

> 这些技术优化的是执行效率，不是模型的"语义本质"。模型本质上做的事情仍然是：基于已有上下文，反复预测下一个 token。

## 4. 系统职责划分

### 推理引擎 / serving 系统负责

- 接 HTTP 请求
- Tokenization / 输入处理
- 调度 batching
- 管理 KV cache
- 协调 GPU worker
- 流式返回结果
- 做一部分采样与系统优化

### LLM 模型本体负责

- 对 input_ids 做 embedding
- 经过多层 Transformer block 的 self-attention 和 feed-forward network
- 输出 logits（"下一个 token 的分数分布"）

> 推理引擎决定"怎么高效地跑"，模型决定"到底生成什么"。前者偏"编排与优化"，后者偏"语义计算与内容生成"。

## 5. 完整链路总结

一个 LLM 请求的推理过程，本质上是：

1. 先把 prompt 模板化并 token 化
2. 经由推理服务调度进入 GPU
3. 模型通过 embedding 和多层 Transformer block 并行完成 prefill
4. 建立上下文表示和 KV cache
5. 进入 decode 循环，基于历史缓存逐 token 执行注意力、前馈网络和采样
6. 直到生成结束，再把 token 序列反解码成文本返回

这条链路同时体现了：

- Transformer 的计算机制
- 自回归生成范式
- 现代推理系统在 batching、缓存和 attention kernel 上的工程优化