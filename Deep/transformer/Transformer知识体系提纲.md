# Transformer 知识体系核心提纲

这份提纲旨在为您提供一个关于 Transformer 及其相关技术的结构化学习路径。它将我们创建的所有笔记组织成一个从基础到前沿的逻辑体系，帮助您更好地理解各个知识点之间的联系。

---

### 第一部分：Transformer 核心架构与基石

*本部分涵盖了构成 Transformer 模型的每一个基础组件，理解这些是掌握一切后续变体的基础。*

1.  **核心交互机制**
    *   [自注意力机制 (Self-Attention)](自注意力机制.md) - Transformer 的灵魂，解释了模型如何权衡序列中不同部分的重要性。
    *   [QKV矩阵计算 (QKV Matrix Calculation)](QKV矩阵计算.md) - 注意力机制的第一步，如何从单一输入得到三个不同的角色表示。
    *   [注意力分数缩放 (Attention Score Scaling)](注意力分数缩放.md) - 解释了为何要除以 `√d_k`，这是保证训练稳定的关键细节。

2.  **增强与扩展**
    *   [多头注意力 (Multi-Head Attention)](Deep/transformer/多头注意力.md) - 为何要将注意力“分而治之”，以及它如何让模型从不同角度理解信息。
    *   [FFN层 (Feed-Forward Network)](FFN层.md) - 注意力之外的另一个核心层，负责对信息进行非线性加工。

3.  **支撑结构**
    *   [位置编码 (Positional Encoding)](Deep/transformer/位置编码.md) - 解决注意力机制无法感知顺序的问题，赋予模型时序概念。
    *   [残差连接 (Residual Connection)](残差连接.md) - 构建深度模型的关键，解决了梯度消失和网络退化问题。
    *   [层归一化 (Layer Normalization)](层归一化.md) - 稳定每层输入的分布，加速并稳定训练过程。

---

### 第二部分：模型训练、解码与推理

*本部分关注 Transformer 如何在训练中学习，以及在推理时如何生成内容，并探讨了两者之间的关键差异。*

1.  **解码器的工作原理**
    *   [因果掩码 (Causal Mask)](因果掩码.md) - 确保 Decoder 在生成时“不看未来”的关键机制。
    *   [自回归生成 (Autoregressive Generation)](自回归生成.md) - 推理时逐词生成文本的标准工作模式。
    *   [Teacher Forcing](Teacher_Forcing.md) - 训练时加速收敛的“作弊”技巧。

2.  **训练与推理的对比**
    *   [Transformer核心问题深度辨析.md](Transformer核心问题深度辨析.md) - (见 Q4) 深入探讨训练与推理在 Decoder 输入上的根本区别。
    *   [KV Cache](KV_Cache.md) - 大幅加速自回归推理的核心优化技术，解决了推理慢的瓶颈。

---

### 第三部分：训练技巧与正则化

*本部分介绍了一系列用于稳定训练过程、防止模型过拟合、提升泛化能力的常用技术。*

*   [学习率预热 (Warmup)](学习率预热.md) - 在训练初期稳定模型的学习步伐。
*   [梯度裁剪 (Gradient Clipping)](梯度裁剪.md) - 防止梯度爆炸，保证训练过程不“失控”。
*   [Dropout应用](Dropout应用.md) - 经典的正则化方法，防止模型“死记硬背”。
*   [标签平滑 (Label Smoothing)](标签平滑.md) - 防止模型对预测过于自信，提升泛化能力。

---

### 第四部分：主流 Transformer 模型家族

*本部分概述了基于 Transformer 架构衍生的、在人工智能领域产生巨大影响的几个里程碑式模型。*

1.  **NLP 领域的巨头**
    *   [BERT](Deep/transformer/BERT.md) - Encoder-only 架构，开启了 NLU（自然语言理解）的预训练新范式。
    *   [GPT](GPT.md) - Decoder-only 架构，现代 LLM（大语言模型）的直系祖先，专注于 NLG（自然语言生成）。
    *   [T5](Deep/transformer/T5.md) - Encoder-Decoder 架构，提出了“万物皆可 Text-to-Text”的统一框架。

2.  **跨领域的延伸**
    *   [视觉Transformer (ViT)](视觉Transformer.md) - 将 Transformer 成功应用于计算机视觉领域的开创性工作。
    *   [跨模态模型](跨模态模型.md) - 如 CLIP，连接文本与图像，是多模态大模型的基础。

---

### 第五部分：模型优化与效率提升

*本部分聚焦于如何让庞大的 Transformer 模型变得更小、更快、更省资源，是模型走向实际应用的关键技术。*

1.  **降低计算复杂度**
    *   [稀疏注意力 (Sparse Attention)](稀疏注意力.md) - 针对注意力 `O(N²)` 瓶颈的优化，是处理长序列的核心思路。

2.  **模型压缩**
    *   [模型量化 (Model Quantization)](模型量化.md) - 通过降低数值精度来压缩模型，提升推理速度。
    *   [知识蒸馏 (Knowledge Distillation)](Deep/transformer/知识蒸馏.md) - “教师”教“学生”，将大模型知识迁移到小模型。
    *   [剪枝 (Pruning)](剪枝.md) - 移除模型中冗余的参数或结构。

3.  **参数高效微调 (PEFT)**
    *   [低秩适配 (LoRA)](低秩适配.md) - 只训练极少量“适配器”参数，即可高效微调整个大模型。

---

### 第六部分：前沿架构与未来展望

*本部分探讨了最新出现的、旨在挑战 Transformer 地位的新架构，并对未来的发展趋势进行了展望。*

*   [MoE架构 (Mixture of Experts)](MoE架构.md) - “混合专家”模式，在控制计算量的前提下，极大地扩展模型参数量。
*   [RetNet (Retentive Network)](RetNet.md) - 结合 RNN 与 Transformer 优点，试图实现“鱼与熊掌兼得”。
*   [Mamba](Mamba.md) - 基于状态空间模型，在长序列任务上展现出强大性能和效率的新星。
*   [Transformer核心问题深度辨析.md](Transformer核心问题深度辨析.md) - (见 Q11) 探讨 Transformer 是否会被 Mamba 等新架构取代。

---

### 附录：综合问题辨析

*   [Transformer核心问题深度辨析.md](Transformer核心问题深度辨析.md) - 该文件集中回答了多个关于 Transformer 的“为什么”和“是什么”的对比性问题，建议在学习完相应的基础模块后阅读，以加深理解。
*   [Transformer比RNN更适合处理长序列的原因.md](Transformer比RNN更适合处理长序列的原因.md) - 深入剖析了 Transformer 相比于其前身 RNN 的根本优势所在。
