# 视觉 Transformer (Vision Transformer, ViT)

## 1. 实现原理

视觉 Transformer（Vision Transformer, ViT）是 Google 在 2020 年提出的一个模型，它首次证明了，在几乎不改动标准 Transformer Encoder 架构的情况下，可以直接将其应用于图像分类任务，并在大规模数据集上取得超越传统卷积神经网络（CNN）的性能。

ViT 的核心思想是：**将图像视为一个由“图像块”（Image Patches）组成的序列，然后用标准的 Transformer Encoder 来处理这个序列。**

**实现流程：**

1.  **图像分块 (Image Patching):**
    *   首先，将输入的二维图像（例如，`224 x 224` 像素）分割成一系列固定大小、不重叠的二维小方块（Patches）。例如，如果每个 patch 的大小是 `16 x 16` 像素，那么一张 `224 x 224` 的图像就会被分割成 `(224/16) * (224/16) = 14 * 14 = 196` 个 patches。

2.  **块线性投影 (Patch Linear Projection):**
    *   对于 Transformer 来说，它的输入应该是一个一维的 token 嵌入序列。因此，需要将每个二维的 patch“压平”（flatten）成一个一维向量（例如，`16 * 16 * 3 = 768` 维，3 代表 RGB 通道）。
    *   然后，通过一个可学习的线性投影层（相当于一个全连接层），将这个压平后的向量映射到模型所需的嵌入维度 `d_model`（例如，768 维）。这个结果被称为 **patch embedding**。

3.  **添加 [CLS] Token:**
    *   与 BERT 的做法类似，在 patch embedding 序列的最前面，人为地添加一个可学习的 `[CLS]` (Classification) token。这个 `[CLS]` token 对应的最终输出向量，将被用作整个图像的聚合表示，以进行最终的分类。

4.  **添加位置编码 (Position Embeddings):**
    *   由于 Transformer 本身不感知序列的顺序，我们必须为每个 patch embedding（包括 `[CLS]` token）添加位置编码，以保留图像块的空间位置信息。ViT 中使用的是可学习的一维位置编码。

5.  **送入标准 Transformer Encoder:**
    *   将融合了位置信息和内容信息的 patch embedding 序列，送入一个标准的多层 Transformer Encoder（由多头自注意力层和 FFN 层组成）。
    *   Encoder 会像处理文本序列一样，对这些图像块进行全局的自注意力计算，捕捉它们之间的依赖关系。

6.  **分类头 (Classification Head):**
    *   在经过多层 Encoder 处理后，取出 `[CLS]` token 对应的最终输出向量。
    *   将这个向量送入一个简单的多层感知机（MLP Head），进行最终的分类预测。

---

## 2. 所解决的问题

ViT 的出现，挑战并部分解决了传统 CNN 架构的一些固有问题：

1.  **CNN 的归纳偏置 (Inductive Bias) 限制:** CNN 的核心是**局部性（Locality）**和**平移不变性（Translation Invariance）**。这使得 CNN 在数据量不足时也能学得很好，但同时也限制了它捕捉**全局长距离依赖**的能力。例如，要关联图像左上角和右下角的两个物体，信息需要通过很多层卷积才能传递到。ViT 的自注意力机制则可以一步到位地捕捉任意两个图像块之间的关系，具有更强的全局建模能力。

2.  **架构统一的可能性:** ViT 成功地将 NLP 领域最主流的 Transformer 架构应用到了 CV 领域，打破了两个领域长期以来模型架构的壁垒，为发展通用的、可以同时处理文本、图像等多种模态的“多模态大模型”铺平了道路。

3.  **可扩展性 (Scalability):** ViT 展现出了比 CNN 更强的可扩展性。实验表明，当训练数据量足够大时（例如，Google 内部的 JFT-300M 数据集），ViT 的性能可以超越当时最顶尖的 CNN 模型。这说明 Transformer 架构能够更有效地从海量数据中获益。

---

## 3. 核心代码

下面是一个使用 PyTorch 实现的简化版 ViT 模型，以帮助理解其核心结构。

```python
import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, 
                 d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        super().__init__()
        
        # 1. 计算序列长度
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1 # +1 for [CLS] token

        # 2. Patch 投影层
        patch_dim = 3 * patch_size * patch_size # 3 for RGB channels
        self.patch_projection = nn.Linear(patch_dim, d_model)

        # 3. 定义 [CLS] token 和位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, d_model))

        # 4. 标准 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x): # x shape: [batch, 3, H, W]
        # a. 图像分块和压平
        # (使用 unfold 或 einops 等库可以更高效地实现)
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(x.size(0), -1, 3 * patch_size * patch_size)
        
        # b. 线性投影
        patch_embeddings = self.patch_projection(patches)
        
        # c. 添加 [CLS] token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # d. 添加位置编码
        embeddings += self.pos_embedding
        
        # e. 送入 Encoder
        encoder_output = self.transformer_encoder(embeddings)
        
        # f. 取出 [CLS] token 的输出进行分类
        cls_output = encoder_output[:, 0]
        logits = self.mlp_head(cls_output)
        
        return logits
```

---

## 4. 实际工程中的应用

ViT 的出现彻底改变了计算机视觉领域的格局，催生了大量的后续研究和应用。

*   **图像分类:** ViT 及其变体（如 DeiT, Swin Transformer）在各大图像分类基准上都取得了顶尖的性能。
*   **目标检测与分割:** 后续的工作将 ViT 的思想扩展到了更复杂的视觉任务，如目标检测和语义分割。例如，Swin Transformer 通过引入层级化的滑动窗口注意力，使得 ViT 能够像 CNN 一样高效地处理多尺度的视觉特征。
*   **多模态学习:** ViT 是构建现代多模态大模型（如 CLIP, DALL-E, Flamingo）的基石。它提供了一种将图像信息“翻译”成 Transformer 所能理解的序列化语言的有效方式，使得统一处理图像和文本成为可能。
*   **医学影像分析:** 在 CT、MRI 等医学影像的分析中，ViT 的全局建模能力可以帮助模型发现传统 CNN 难以捕捉的病理特征。

尽管 ViT 在大规模数据上表现优异，但它也存在**数据饥渴（data-hungry）**的问题。在数据量较小（如 ImageNet-1k）的情况下，其性能通常不如精心设计的 CNN。为了解决这个问题，后续的 DeiT 等模型通过引入更先进的训练策略和知识蒸馏，使得 ViT 在中等规模的数据集上也能取得很好的效果。
