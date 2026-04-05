# 09_RAG与检索增强

## RAG系统详解

# RAG系统详解

## 1. RAG基本原理

### 1.1 核心概念

RAG（Retrieval-Augmented Generation）结合了**检索（Retrieval）**和**生成（Generation）**两个模块，利用外部知识库增强生成模型的输出质量。

#### 核心思想
1. **检索**：从大规模文档库中检索与输入相关的信息
2. **生成**：基于检索到的内容，生成更准确、可靠的回答

### 1.2 工作流程

RAG分为两个主要阶段：

#### 检索阶段（Retrieval Phase）
给定输入query $q$，使用检索模型从外部知识库 $D = \{d_1, d_2, ..., d_N\}$ 中找出最相关的top-$k$文档：

$$\text{Retrieve}(q, D) = \text{top-}k \text{ ranked } \{d_i\}_{i=1}^k$$

#### 生成阶段（Generation Phase）
将检索到的文档 $d_1, ..., d_k$ 与原始query $q$ 拼接，输入到生成模型生成最终答案：

$$p(y | q, d_1, ..., d_k) = \text{LM}([q; d_1; ...; d_k])$$

### 1.3 数学表示

给定query $q$，RAG的生成概率可分解为：

$$p(y | q) = \sum_{d \in D} \underbrace{p(d | q)}_{\text{检索阶段}} \cdot \underbrace{p(y | q, d)}_{\text{生成阶段}}$$

其中：
- $p(d | q)$ 由检索模型计算
- $p(y | q, d)$ 由生成模型计算

## 2. 检索方法详解

### 2.1 稀疏检索（Sparse Retrieval）

#### TF-IDF
基于词频统计匹配query和文档：

- **TF（词频）**：$\text{TF}(t, d) = \frac{\text{词 } t \text{ 在 } d \text{ 中的出现次数}}{\text{文档 } d \text{ 的总词数}}$
- **IDF（逆文档频率）**：$\text{IDF}(t) = \log \frac{\text{总文档数}}{\text{包含词 } t \text{ 的文档数}}$
- **相似度计算**：$\text{score}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}$

#### BM25
对TF-IDF进行加权，抑制高频词的影响：

$$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

其中 $k_1 \in [1.2, 2.0]$, $b=0.75$ 为超参数。

### 2.2 稠密检索（Dense Retrieval）

#### DPR（Dense Passage Retrieval）
使用双编码器架构：

- **Query编码器**：$E_Q(q) \in \mathbb{R}^d$
- **Passage编码器**：$E_P(d) \in \mathbb{R}^d$
- **相似度计算**：$\text{score}(q, d) = E_Q(q)^T E_P(d)$

#### 训练目标
对比学习，最大化正样本分数，最小化负样本分数：

$$\mathcal{L} = -\log \frac{e^{\text{score}(q, d^+)}}{e^{\text{score}(q, d^+)} + \sum_{d^-} e^{\text{score}(q, d^-)}}$$

#### ANCE（Approximate Nearest Neighbor Negative Contrastive Learning）
动态生成困难负样本：

1. 初步训练基础DPR模型
2. 使用该模型检索每个query的top相关文档
3. 从中采样非正样本作为困难负例
4. 用新负例重新训练模型

#### ColBERT（Contextualized Late Interaction）
对query和文档的每个token分别编码：

$$\text{score}(q, d) = \sum_{t_q \in q} \max_{t_d \in d} \mathbf{v}_{t_q}^T \mathbf{v}_{t_d}$$

### 2.3 混合检索（Hybrid Retrieval）

结合稀疏检索和稠密检索的优点：

- **分数融合**：$\text{score}_{\text{hybrid}} = \alpha \cdot \text{score}_{\text{BM25}} + (1-\alpha) \cdot \text{score}_{\text{DPR}}$
- **级联检索**：先用BM25粗筛，再用DPR精排

## 3. RAG与传统生成模型的对比

| 特性 | RAG | 传统生成模型（GPT） |
|------|-----|-------------------|
| **知识来源** | 外部知识库（动态检索） | 仅依赖预训练参数（静态知识） |
| **可解释性** | 高（可溯源检索文档） | 低（黑盒生成） |
| **知识更新方式** | 仅需更新检索库 | 需重新微调或预训练 |
| **长尾/最新知识** | 表现更好（依赖检索） | 可能过时或错误 |
| **计算成本** | 较高（检索+生成） | 较低（仅生成） |
| **应用场景** | 开放域QA、事实性任务 | 通用文本生成 |

## 4. RAG系统实现

### 4.1 基础架构

```python
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, retriever_model, generator_model, documents):
        self.retriever = SentenceTransformer(retriever_model)
        self.generator = AutoModel.from_pretrained(generator_model)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.documents = documents
        
        # 构建文档索引
        self.doc_embeddings = self.retriever.encode(documents)
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings.astype('float32'))
    
    def retrieve(self, query, top_k=5):
        """检索相关文档"""
        query_embedding = self.retriever.encode([query])
        scores, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
    
    def generate(self, query, retrieved_docs):
        """基于检索结果生成答案"""
        context = " ".join(retrieved_docs)
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.generator.generate(**inputs, max_length=100, num_beams=5)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def answer(self, query, top_k=5):
        """完整的RAG流程"""
        retrieved_docs = self.retrieve(query, top_k)
        answer = self.generate(query, retrieved_docs)
        return answer, retrieved_docs
```

### 4.2 高级实现

#### 多路检索
```python
class MultiRetrieverRAG:
    def __init__(self, retrievers, generator_model):
        self.retrievers = retrievers  # 多个检索器
        self.generator = AutoModel.from_pretrained(generator_model)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
    
    def retrieve_multiple(self, query, top_k=5):
        """使用多个检索器"""
        all_docs = []
        for retriever in self.retrievers:
            docs = retriever.retrieve(query, top_k)
            all_docs.extend(docs)
        
        # 去重和重排序
        unique_docs = list(set(all_docs))
        return self.rerank(query, unique_docs, top_k)
    
    def rerank(self, query, docs, top_k):
        """重排序检索结果"""
        # 使用更复杂的重排序模型
        scores = []
        for doc in docs:
            score = self.compute_relevance_score(query, doc)
            scores.append(score)
        
        # 按分数排序
        sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return sorted_docs[:top_k]
```

#### 实时更新
```python
class DynamicRAG:
    def __init__(self, retriever_model, generator_model):
        self.retriever = SentenceTransformer(retriever_model)
        self.generator = AutoModel.from_pretrained(generator_model)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.documents = []
        self.doc_embeddings = None
        self.index = None
    
    def add_documents(self, new_docs):
        """动态添加文档"""
        self.documents.extend(new_docs)
        new_embeddings = self.retriever.encode(new_docs)
        
        if self.doc_embeddings is None:
            self.doc_embeddings = new_embeddings
        else:
            self.doc_embeddings = np.vstack([self.doc_embeddings, new_embeddings])
        
        # 重建索引
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings.astype('float32'))
    
    def remove_documents(self, doc_ids):
        """删除文档"""
        # 实现文档删除逻辑
        pass
```

## 5. 检索优化策略

### 5.1 负样本策略

#### 困难负样本挖掘
```python
def mine_hard_negatives(queries, positive_docs, retriever, top_k=100):
    """挖掘困难负样本"""
    hard_negatives = []
    
    for query in queries:
        # 检索top-k文档
        candidates = retriever.retrieve(query, top_k)
        
        # 过滤掉正样本
        negatives = [doc for doc in candidates if doc not in positive_docs[query]]
        
        # 选择最困难的负样本（分数最高的非正样本）
        hard_negatives.extend(negatives[:10])
    
    return hard_negatives
```

#### 对比学习训练
```python
def contrastive_loss(query_emb, pos_emb, neg_embs, temperature=0.1):
    """对比学习损失"""
    # 计算正样本分数
    pos_score = torch.cosine_similarity(query_emb, pos_emb, dim=1)
    
    # 计算负样本分数
    neg_scores = torch.cosine_similarity(query_emb.unsqueeze(1), neg_embs, dim=2)
    
    # 对比学习损失
    logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long)
    
    return torch.nn.functional.cross_entropy(logits, labels)
```

### 5.2 检索效率优化

#### 近似最近邻搜索
```python
import faiss

def build_hnsw_index(embeddings, M=32, efConstruction=200):
    """构建HNSW索引"""
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = efConstruction
    index.add(embeddings.astype('float32'))
    return index

def build_ivf_index(embeddings, nlist=100):
    """构建IVF索引"""
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    return index
```

#### 批量检索
```python
def batch_retrieve(queries, retriever, batch_size=32, top_k=5):
    """批量检索"""
    all_results = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_embeddings = retriever.encode(batch_queries)
        
        # 批量搜索
        scores, indices = retriever.index.search(batch_embeddings, top_k)
        
        for j, (score, idx) in enumerate(zip(scores, indices)):
            results = [(retriever.documents[i], s) for i, s in zip(idx, score)]
            all_results.append(results)
    
    return all_results
```

## 6. 生成优化策略

### 6.1 上下文优化

#### 文档重排序
```python
def rerank_documents(query, docs, reranker_model):
    """使用重排序模型优化文档顺序"""
    pairs = [(query, doc) for doc in docs]
    scores = reranker_model.predict(pairs)
    
    # 按分数排序
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs
```

#### 上下文压缩
```python
def compress_context(docs, max_length=1000):
    """压缩上下文长度"""
    compressed_docs = []
    current_length = 0
    
    for doc in docs:
        if current_length + len(doc) <= max_length:
            compressed_docs.append(doc)
            current_length += len(doc)
        else:
            # 截断文档
            remaining = max_length - current_length
            if remaining > 100:  # 至少保留100个字符
                compressed_docs.append(doc[:remaining])
            break
    
    return compressed_docs
```

### 6.2 生成策略优化

#### 多候选生成
```python
def generate_multiple_candidates(query, docs, generator, num_candidates=5):
    """生成多个候选答案"""
    candidates = []
    
    for _ in range(num_candidates):
        # 随机选择部分文档
        selected_docs = random.sample(docs, min(3, len(docs)))
        context = " ".join(selected_docs)
        
        # 生成答案
        answer = generator.generate(query, context)
        candidates.append(answer)
    
    return candidates

def select_best_answer(query, candidates, selector_model):
    """选择最佳答案"""
    scores = []
    for candidate in candidates:
        score = selector_model.score(query, candidate)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return candidates[best_idx]
```

## 7. 评估指标

### 7.1 检索评估

#### 召回率（Recall）
$$Recall@k = \frac{|\text{相关文档} \cap \text{检索到的top-k文档}|}{|\text{相关文档}|}$$

#### 精确率（Precision）
$$Precision@k = \frac{|\text{相关文档} \cap \text{检索到的top-k文档}|}{k}$$

#### NDCG（Normalized Discounted Cumulative Gain）
$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

其中：
$$DCG@k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

### 7.2 生成评估

#### ROUGE分数
```python
from rouge_score import rouge_scorer

def compute_rouge_scores(predictions, references):
    """计算ROUGE分数"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        for metric in scores:
            scores[metric].append(score[metric].fmeasure)
    
    return {metric: np.mean(values) for metric, values in scores.items()}
```

#### BLEU分数
```python
from nltk.translate.bleu_score import sentence_bleu

def compute_bleu_scores(predictions, references):
    """计算BLEU分数"""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        score = sentence_bleu([ref_tokens], pred_tokens)
        scores.append(score)
    
    return np.mean(scores)
```

## 8. 实际应用场景

### 8.1 开放域问答
```python
class OpenDomainQA:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def answer_question(self, question):
        # 检索相关文档
        docs = self.retriever.retrieve(question, top_k=5)
        
        # 生成答案
        answer = self.generator.generate(question, docs)
        
        return answer, docs
```

### 8.2 事实核查
```python
class FactChecker:
    def __init__(self, retriever, generator, verifier):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
    
    def check_fact(self, claim):
        # 检索支持证据
        evidence = self.retriever.retrieve(claim, top_k=10)
        
        # 生成验证结果
        verification = self.verifier.verify(claim, evidence)
        
        return verification, evidence
```

### 8.3 对话系统
```python
class ConversationalRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.conversation_history = []
    
    def respond(self, user_input):
        # 构建上下文感知的查询
        context_query = self.build_context_query(user_input)
        
        # 检索相关文档
        docs = self.retriever.retrieve(context_query, top_k=3)
        
        # 生成回复
        response = self.generator.generate(context_query, docs)
        
        # 更新对话历史
        self.conversation_history.append((user_input, response))
        
        return response
    
    def build_context_query(self, current_input):
        """构建包含对话历史的查询"""
        if len(self.conversation_history) == 0:
            return current_input
        
        # 取最近几轮对话作为上下文
        recent_context = self.conversation_history[-3:]
        context_text = " ".join([f"User: {u}\nAssistant: {a}" for u, a in recent_context])
        
        return f"Context: {context_text}\nCurrent: {current_input}"
```

## 9. 挑战与未来方向

### 9.1 当前挑战

1. **检索噪声**：无关文档影响生成质量
2. **计算开销**：实时检索+生成成本高
3. **知识更新**：如何实时更新检索库
4. **多模态**：如何处理文本、图像、视频等多模态信息

### 9.2 改进方向

1. **端到端训练**：联合优化检索器和生成器
2. **动态检索**：根据生成过程动态调整检索策略
3. **知识图谱集成**：结合结构化知识
4. **多跳推理**：支持多步推理的检索增强

## 总结

RAG系统通过结合检索和生成，显著提升了生成模型的**事实性**和**可解释性**：

1. **检索模块**：提供准确的相关信息
2. **生成模块**：基于检索结果生成高质量答案
3. **优化策略**：提升检索精度和生成质量
4. **实际应用**：开放域问答、事实核查、对话系统等

RAG是当前NLP领域的重要研究方向，在知识密集型任务中表现出色。 

