# GPT (Generative Pre-trained Transformer)

## 1. 实现原理

GPT（生成式预训练 Transformer）是由 OpenAI 在 2018 年提出的一个语言模型系列。与 BERT 专注于“理解”文本不同，GPT 的核心目标是**“生成”**文本。它通过大规模的无监督预训练，学习生成连贯、流畅、且符合上下文逻辑的文本。

**架构：只使用 Transformer Decoder**

GPT 的模型架构与 BERT 恰好相反，它只使用了 Transformer 的 **Decoder** 部分。一个标准的 GPT 模型由多层（例如，GPT-1 有 12 层）的 Transformer Decoder 堆叠而成。这种架构天生就适合于自回归（Autoregressive）的文本生成任务。

**核心机制：因果掩码 (Causal Mask)**

由于使用了 Decoder 架构，GPT 的自注意力层中必须使用**因果掩码**。这确保了模型在预测第 `t` 个 token 时，只能看到前面的 `t-1` 个 token，而不能“偷看”未来的信息。这种严格的从左到右的单向信息流，是 GPT 能够作为“语言模型”进行生成的基础。

**预训练任务：标准语言模型 (Standard Language Model)**

GPT 的预训练任务非常纯粹和经典，就是**自回归语言建模**。给定一个无标签的文本序列 `(t_1, t_2, ..., t_n)`，模型的目标是最大化这个序列出现的概率。根据链式法则，这个联合概率可以分解为一系列条件概率的乘积：

`P(t_1, ..., t_n) = Π P(t_i | t_1, ..., t_{i-1})`

也就是说，模型在训练时，不断地学习根据已经出现的上文，来预测下一个最有可能出现的词。这本质上就是一种**“Teacher Forcing”**的应用，模型在每一步都被输入了真实的文本序列。

**GPT 与 BERT 的核心区别：**

| 特性 | GPT (Decoder-only) | BERT (Encoder-only) |
| :--- | :--- | :--- |
| **架构** | Transformer Decoder | Transformer Encoder |
| **信息流** | 单向 (Unidirectional)，从左到右 | 双向 (Bidirectional) |
| **核心机制** | 因果掩码 (Causal Mask) | 掩码语言模型 (MLM) |
| **预训练任务** | 预测下一个词 | 预测被掩盖的词 |
| **强项** | 文本生成 (NLG) | 文本理解 (NLU) |

---

## 2. 所解决的问题

GPT 模型主要解决了以下问题：

1.  **高效的文本生成能力:** GPT 的 Decoder-only 架构和自回归预训练任务，使其天然地成为一个强大的文本生成器。它能够生成语法正确、语义连贯、风格一致的长文本。

2.  **零样本/少样本学习 (Zero/Few-shot Learning):** 这是 GPT 系列模型（尤其是从 GPT-2 开始）展现出的惊人能力。由于在海量、多样化的数据上进行了预训练，模型不仅学到了语言规则，还学到了关于世界的大量事实性知识和任务模式。因此，对于很多新任务，我们甚至**不需要**为它准备特定的训练数据进行微调，只需要在**提示 (Prompt)** 中给出任务的描述或几个示例，模型就能“理解”任务要求并直接给出结果。这极大地拓宽了语言模型的应用场景。

3.  **统一的生成式预训练框架:** GPT 证明了通过一个统一的、简单的生成式预训练目标，就可以构建一个能够适应多种不同任务的通用模型，为后续的大型语言模型（LLM）发展奠定了基础。

---

## 3. 核心代码

与 BERT 类似，使用 GPT 模型最常见的方式是通过 Hugging Face `transformers` 库。下面以 GPT-2 为例，展示如何使用预训练好的模型进行文本生成。

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载预训练好的 Tokenizer 和模型
# 'gpt2' 是一个基础版的英文 GPT-2 模型
# GPT2LMHeadModel 表示这是一个带有语言模型头 (用于生成) 的 GPT-2 模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 2. 准备输入文本 (Prompt)
# pad_token_id is required for batch generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "The future of AI is"

# 3. 使用 Tokenizer 对文本进行编码
encoded_input = tokenizer(prompt, return_tensors='pt')

# 4. 使用模型的 .generate() 方法进行生成
# .generate() 是一个高度封装的函数，内置了自回归生成、KV Cache、
# 多种搜索策略 (贪心、束搜索、采样) 等复杂逻辑。
output_sequences = model.generate(
    input_ids=encoded_input['input_ids'],
    attention_mask=encoded_input['attention_mask'],
    max_length=50,  # 生成的最大长度
    num_return_sequences=1, # 生成几个不同的序列
    do_sample=True, # 使用采样策略，增加多样性
    top_k=50, # Top-K 采样
    top_p=0.95, # Top-P (Nucleus) 采样
    pad_token_id=tokenizer.pad_token_id
)

# 5. 解码生成的序列
decoded_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print(decoded_text)
# 可能的输出: 
# The future of AI is not in the hands of a few companies, but in the hands of the people.
# The future of AI is bright, but it will be a long time before we see the full potential of the technology.
```

---

## 4. 实际工程中的应用

GPT 系列模型，特别是其后续发展出的**大型语言模型 (LLM)**，已经成为当前人工智能领域最受关注和应用最广泛的技术。

*   **对话式 AI / 聊天机器人:** ChatGPT 是 GPT 应用最成功的典范，它能够进行流畅、自然、且富有知识的多轮对话。
*   **内容创作:** 用于撰写文章、博客、营销文案、邮件、诗歌等各种文本内容。
*   **代码辅助与生成:** 能够根据自然语言描述生成代码、解释代码、修复 bug，如 GitHub Copilot。
*   **智能问答与知识检索:** 作为强大的知识库，可以直接回答用户提出的各种问题。
*   **情感分析与文本摘要:** 通过精心设计的 Prompt，可以让 GPT 完成各种传统的 NLU 任务。

从 GPT-1 到 GPT-2，再到 GPT-3 和 GPT-4，模型规模的不断扩大和训练数据的不断增加，使得模型的生成能力和“智能”水平发生了质的飞跃，开启了当前的生成式 AI 时代。几乎所有我们今天看到的 LLM，都可以看作是 GPT 架构的直接继承者和发扬者。
