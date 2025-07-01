### **BLEU（Bilingual Evaluation Understudy）详解**

BLEU 是一种用于评估**机器翻译（或其他文本生成任务）质量**的自动评价指标，通过比较生成文本（Candidate）与参考文本（Reference）的相似度来打分。其核心思想是衡量生成文本在**词汇、词序、长度**等方面与人工参考译文的匹配程度。

***

## **1. BLEU 的作用**

*   **自动化评估**：替代人工打分，快速比较不同模型或算法的翻译质量。
*   **多参考支持**：允许使用多个参考译文（人类翻译的多个版本）提高评估鲁棒性。
*   **广泛应用**：不仅用于机器翻译，还可用于文本摘要、对话生成等任务。

***

## **2. BLEU 的原理**

BLEU 的计算基于两个核心部分：**n-gram 精度** 和 **长度惩罚**。

### **(1) n-gram 精度（Modified Precision）**

*   **定义**：统计生成文本中与参考文本匹配的 n-gram（连续n个词）的比例。
*   **改进**：为避免重复词夸大分数，对每个n-gram的计数进行**截断**（不超过参考文本中的最大出现次数）。\
    **公式**：

```math
    P\_n = \frac{\sum\_{\text{n-gram} \in C} \min(\text{Count}*{\text{reference}}(n\text{-gram}), \text{Count}*{\text{candidate}}(n\text{-gram}))}{\sum\_{\text{n-gram} \in C} \text{Count}\_{\text{candidate}}(n\text{-gram})}
```

    *   (C) 是生成文本，(P_n) 是n-gram精度（通常计算 (n=1,2,3,4)）。

### **(2) 长度惩罚（Brevity Penalty, BP）**

*   **问题**：短文本可能因高n-gram精度而得分虚高。
*   **解决**：若生成文本长度（(l\_c)）小于参考文本最短长度（(l\_r)），则惩罚得分。\
    **公式**：

```math
    BP =
    \begin{cases}
    1 & \text{if } l\_c \ge l\_r \\
    e^{1 - l\_r / l\_c} & \text{if } l\_c < l\_r
    \end{cases}
```

### **(3) 综合 BLEU 分数**

将不同n-gram的精度几何平均后乘以长度惩罚：

```math
BLEU = BP \cdot \exp\left(\sum\_{n=1}^N w\_n \log P\_n\right)
```

*   默认权重 `$(w\_n = \frac{1}{N})$`（通常 (N=4)，即1-gram到4-gram权重相同）。

***

## **3. BLEU 的使用方法**

### **(1) 安装工具**

```bash
pip install nltk  # 或使用 sacrebleu（更标准化）
```

### **(2) 代码示例（Python）**

#### **方法1：NLTK 实现**

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# 单句评估
reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]  # 参考译文（可多组）
candidate = ['the', 'cat', 'sat', 'on', 'the', 'mat']   # 生成文本
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)  # 输出0.59（满分1.0）

# 语料库评估（多句子）
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]  # 每个句子多组参考
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
```

#### **方法2：SacreBLEU（推荐）**

```bash
pip install sacrebleu
```

```python
import sacrebleu

references = [["The cat is on the mat."]]  # 注意：输入为字符串，而非分词列表
candidate = "The cat sat on the mat."
score = sacrebleu.sentence_bleu(candidate, references)
print(score.score)  # 输出46.51（SacreBLEU默认缩放为0~100）
```

***

## **4. BLEU 的优缺点**

### **优点**

*   **自动化**：无需人工参与，适合大规模评估。
*   **可重复性**：结果稳定，适合论文实验对比。

### **缺点**

*   **语义盲区**：只关注表面匹配，忽略同义词和语义一致性。
*   **短文本偏差**：对短文本打分偏高（即使漏译关键信息）。
*   **依赖分词**：不同分词工具可能导致分数差异。

***

## **5. 常见问题**

### **Q1：BLEU 分数多少算好？**

*   人类翻译通常得分为 50\~70（SacreBLEU 标准）。
*   但需结合任务：对话生成可能 20\~30 即可，机器翻译需 >40。

### **Q2：与 ROUGE、METEOR 的区别？**

*   **ROUGE**：用于摘要任务，侧重召回率（参考文本中的词是否被覆盖）。
*   **METEOR**：引入同义词和词干匹配，对语义更敏感。

### **Q3：如何提高 BLEU 分数？**

*   增加训练数据。
*   使用更大的模型（如 Transformer）。
*   调整生成长度（避免过短）。

***

## **总结**

*   **BLEU** 通过 n-gram 精度和长度惩罚评估生成文本质量。
*   **用法**：直接调用 `nltk` 或 `sacrebleu` 计算，支持单句和语料库。
*   **注意**：BLEU 需结合人工评估或其他指标（如 TER、BERTScore）综合判断。

## n-gram

我将用**具体数据和类比**详细解释 **n-gram**，并结合 **BLEU 的计算过程**，让你彻底明白它的作用。

***

### **1. n-gram 是什么？**

**n-gram** 是文本中连续的 `n` 个词（或字）的组合。它用于捕捉语言的局部模式（如短语、常见搭配）。

*   **示例句子**：`"I love deep learning"`
    *   **1-gram（unigram）**：`["I", "love", "deep", "learning"]`
    *   **2-gram（bigram）**：`["I love", "love deep", "deep learning"]`
    *   **3-gram（trigram）**：`["I love deep", "love deep learning"]`

***

### **2. BLEU 中 n-gram 的作用**

在 BLEU 中，我们通过统计生成文本和参考文本的 **n-gram 重叠比例** 来衡量质量。

*   **核心思想**：
    *   如果生成文本的 n-gram 和参考文本一致，说明翻译准确。
    *   使用多个 `n`（通常1-4）同时评估**单词选择**和**词序流畅性**。

***

### **3. 具体计算示例**

#### **输入数据**

*   **生成文本（Candidate）**：`"the cat sat on the mat"`
*   **参考文本（Reference）**：`"the cat is on the mat"`

#### **步骤1：计算 n-gram 匹配**

##### **(1) 1-gram 精度**

*   **生成文本的1-gram**：`["the", "cat", "sat", "on", "the", "mat"]`
*   **参考文本的1-gram**：`["the", "cat", "is", "on", "the", "mat"]`
*   **匹配的1-gram**：`"the"`（2次）, `"cat"`（1次）, `"on"`（1次）, `"mat"`（1次）
*   **截断计数**（避免重复词夸大分数）：
    *   `"the"` 在生成文本中出现2次，但在参考文本中仅出现2次 → 取 `min(2,2)=2`
    *   其他词匹配次数均为1。
*   **1-gram精度**：

```math
    P\_1 = \frac{2(\text{the}) + 1(\text{cat}) + 1(\text{on}) + 1(\text{mat})}{6} = \frac{5}{6} \approx 0.83
```

##### **(2) 2-gram 精度**

*   **生成文本的2-gram**：`["the cat", "cat sat", "sat on", "on the", "the mat"]`
*   **参考文本的2-gram**：`["the cat", "cat is", "is on", "on the", "the mat"]`
*   **匹配的2-gram**：`"the cat"`, `"on the"`, `"the mat"`（各1次）
*   **2-gram精度**：

```math
    P\_2 = \frac{3}{5} = 0.6
```

##### **(3) 更高阶 n-gram**

同理计算3-gram、4-gram的精度（本例中无匹配，因此 (P\_3=0, P\_4=0)）。

#### **步骤2：长度惩罚**

*   **生成文本长度（(l\_c)）**：6
*   **参考文本长度（(l\_r)）**：6
*   **长度惩罚（BP）**：

```math
    BP = 1 \quad (\text{因为 } l\_c = l\_r)
```

#### **步骤3：综合 BLEU 分数**

假设权重均匀（(w\_n=0.25)）：

```math
BLEU = 1 \cdot \exp\left(0.25 \cdot \log 0.83 + 0.25 \cdot \log 0.6 + 0.25 \cdot \log 0 + 0.25 \cdot \log 0\right)
```

由于 (\log 0) 无定义，实际中会避免 (P\_n=0)（如设置极小值 ( `$\epsilon$` )），最终分数接近0。

***

### **4. 为什么用多阶 n-gram？**

*   **1-gram**：衡量**单词选择**准确性（如是否用对词）。
*   **2/3/4-gram**：衡量**短语搭配**和**词序流畅性**（如"猫坐在" vs "猫是在"）。
*   **高阶n-gram**：捕捉更长的语法结构（但过高的 `n` 会导致稀疏性问题）。

***

### **5. 可视化对比**

| n-gram | 生成文本                                        | 参考文本                                      | 匹配结果   |
| ------ | ------------------------------------------- | ----------------------------------------- | ------ |
| 1-gram | `the, cat, sat, on, the, mat`               | `the, cat, is, on, the, mat`              | 5/6 匹配 |
| 2-gram | `the cat, cat sat, sat on, on the, the mat` | `the cat, cat is, is on, on the, the mat` | 3/5 匹配 |

***

### **6. 关键点总结**

1.  **n-gram 是连续的词组合**，用于量化文本的局部相似性。
2.  **BLEU 通过多阶 n-gram 精度** 综合评估翻译质量。
3.  **长度惩罚** 防止短文本得分虚高。
4.  **实际应用**：通常计算1-gram到4-gram的几何平均（如 `weights=(0.25,0.25,0.25,0.25)`）。

通过这个例子，你应该能直观理解 n-gram 如何用于 BLEU 计算了！
