# 05_NLP核心任务

## NLP核心任务详解

# NLP核心任务详解

## 1. 文本分类

### 1.1 任务定义

文本分类是将文本分配到预定义类别的任务，如情感分析、主题分类、垃圾邮件检测等。

### 1.2 传统方法

#### TF-IDF + 机器学习
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def traditional_text_classifier():
    """传统文本分类器"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    return pipeline

# 使用示例
classifier = traditional_text_classifier()
classifier.fit(train_texts, train_labels)
predictions = classifier.predict(test_texts)
```

#### Word2Vec + 机器学习
```python
import numpy as np
from gensim.models import Word2Vec

def get_document_vector(text, word2vec_model):
    """获取文档向量（词向量的平均值）"""
    words = text.split()
    word_vectors = []
    
    for word in words:
        if word in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[word])
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1)
```

### 1.3 深度学习方法

#### TextCNN
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1)    # (batch_size, 1, seq_len, embed_dim)
        
        # 卷积操作
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_len-k+1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len-k+1)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_out = conv_out.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(conv_out)
        
        # 拼接所有卷积输出
        x = torch.cat(conv_outputs, 1)  # (batch_size, len(kernel_sizes) * num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

#### BERT分类
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BERTClassifier:
    def __init__(self, model_name='bert-base-chinese', num_classes=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
    
    def predict(self, texts):
        """预测文本类别"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                predictions.append(pred)
        
        return predictions
```

## 2. 命名实体识别（NER）

### 2.1 任务定义

NER是从文本中识别和分类命名实体的任务，如人名、地名、组织名等。

### 2.2 传统方法

#### CRF
```python
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

def word2features(sent, i):
    """提取词特征"""
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True
    
    return features

def sent2features(sent):
    """句子转特征"""
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """句子转标签"""
    return [label for token, postag, label in sent]

# 训练CRF模型
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
crf.fit(X_train, y_train)
```

### 2.3 深度学习方法

#### BiLSTM-CRF
```python
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=1, bidirectional=True)
        
        # 将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF层
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.start_transitions = nn.Parameter(torch.randn(self.tagset_size))
        self.end_transitions = nn.Parameter(torch.randn(self.tagset_size))
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    
    def _forward_alg(self, feats):
        """前向算法计算归一化因子"""
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix['START_TAG']] = 0.
        
        forward_var = init_alphas
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.end_transitions
        alpha = self.log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        """获取LSTM特征"""
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        """计算给定标签序列的分数"""
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix['START_TAG']], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['STOP_TAG'], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        """维特比解码"""
        backpointers = []
        
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix['START_TAG']] = 0
        
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        
        terminal_var = forward_var + self.end_transitions
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix['START_TAG']
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        """负对数似然损失"""
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    
    def forward(self, sentence):
        """前向传播"""
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
```

#### BERT-NER
```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

class BERTNER:
    def __init__(self, model_name='bert-base-chinese', num_labels=9):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    
    def predict(self, text):
        """预测命名实体"""
        self.model.eval()
        
        # 分词
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # 解码预测结果
        entities = self.decode_entities(tokens, predictions[0].tolist())
        return entities
    
    def decode_entities(self, tokens, predictions):
        """解码实体"""
        entities = []
        current_entity = None
        
        for token, pred in zip(tokens, predictions[1:-1]):  # 跳过[CLS]和[SEP]
            if pred == 1:  # B-XXX
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'text': token, 'type': 'PERSON'}
            elif pred == 2:  # I-XXX
                if current_entity:
                    current_entity['text'] += token
            else:  # O
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
```

## 3. 文本摘要

### 3.1 抽取式摘要

#### TextRank
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class TextRank:
    def __init__(self, damping=0.85, min_diff=1e-5, steps=100):
        self.damping = damping
        self.min_diff = min_diff
        self.steps = steps
    
    def extract_summary(self, sentences, top_k=3):
        """提取摘要"""
        # 计算句子相似度矩阵
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # 归一化相似度矩阵
        similarity_matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)
        
        # TextRank算法
        scores = np.ones(len(sentences)) / len(sentences)
        
        for _ in range(self.steps):
            new_scores = (1 - self.damping) + self.damping * similarity_matrix.T.dot(scores)
            diff = abs(new_scores - scores).sum()
            scores = new_scores
            
            if diff <= self.min_diff:
                break
        
        # 选择top-k句子
        top_indices = scores.argsort()[-top_k:][::-1]
        summary = [sentences[i] for i in sorted(top_indices)]
        
        return summary, scores
```

#### BERT抽取式摘要
```python
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

class BERTSummarizer:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    
    def extract_summary(self, text, top_k=3):
        """使用BERT提取摘要"""
        # 分句
        sentences = self.split_sentences(text)
        
        # 计算句子表示
        sentence_embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的表示作为句子表示
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                sentence_embeddings.append(embedding)
        
        sentence_embeddings = torch.stack(sentence_embeddings)
        
        # 计算句子相似度
        similarity_matrix = F.cosine_similarity(
            sentence_embeddings.unsqueeze(1), 
            sentence_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # 计算句子重要性分数
        scores = similarity_matrix.sum(dim=1)
        
        # 选择top-k句子
        top_indices = scores.argsort(descending=True)[:top_k]
        summary = [sentences[i] for i in top_indices]
        
        return summary
    
    def split_sentences(self, text):
        """分句"""
        import re
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
```

### 3.2 生成式摘要

#### Seq2Seq摘要
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(Seq2SeqSummarizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.output.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)
        
        # 编码
        embedded = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # 解码器初始状态
        decoder_hidden = hidden[-2:].mean(0).unsqueeze(0).repeat(2, 1, 1)
        decoder_cell = cell[-2:].mean(0).unsqueeze(0).repeat(2, 1, 1)
        
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            embedded = self.embedding(decoder_input)
            
            # 注意力机制
            attention_weights = F.softmax(
                self.attention(encoder_outputs), dim=1
            )
            context = torch.sum(attention_weights * encoder_outputs, dim=1).unsqueeze(1)
            
            # 解码
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                embedded, (decoder_hidden, decoder_cell)
            )
            
            output = self.output(decoder_output)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
```

## 4. 机器翻译

### 4.1 统计机器翻译

#### 基于短语的翻译
```python
class PhraseBasedMT:
    def __init__(self):
        self.phrase_table = {}
        self.language_model = {}
    
    def build_phrase_table(self, parallel_corpus):
        """构建短语表"""
        for src_sent, tgt_sent in parallel_corpus:
            src_words = src_sent.split()
            tgt_words = tgt_sent.split()
            
            # 提取短语对
            for i in range(len(src_words)):
                for j in range(i + 1, len(src_words) + 1):
                    src_phrase = ' '.join(src_words[i:j])
                    tgt_phrase = ' '.join(tgt_words[i:j])
                    
                    if src_phrase not in self.phrase_table:
                        self.phrase_table[src_phrase] = {}
                    
                    if tgt_phrase not in self.phrase_table[src_phrase]:
                        self.phrase_table[src_phrase][tgt_phrase] = 0
                    
                    self.phrase_table[src_phrase][tgt_phrase] += 1
    
    def translate(self, src_sentence):
        """翻译句子"""
        src_words = src_sentence.split()
        translations = []
        
        i = 0
        while i < len(src_words):
            # 寻找最长匹配的短语
            best_phrase = None
            best_translation = None
            max_len = 0
            
            for j in range(i + 1, len(src_words) + 1):
                phrase = ' '.join(src_words[i:j])
                if phrase in self.phrase_table:
                    for tgt_phrase, count in self.phrase_table[phrase].items():
                        if j - i > max_len:
                            max_len = j - i
                            best_phrase = phrase
                            best_translation = tgt_phrase
            
            if best_translation:
                translations.append(best_translation)
                i += max_len
            else:
                # 如果没有找到短语，逐词翻译
                translations.append(src_words[i])
                i += 1
        
        return ' '.join(translations)
```

### 4.2 神经机器翻译

#### Transformer翻译
```python
from transformers import MarianMTModel, MarianTokenizer

class TransformerMT:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-zh-en'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, text):
        """翻译文本"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=128, num_beams=5)
        
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def batch_translate(self, texts):
        """批量翻译"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=128, num_beams=5)
        
        translations = [self.tokenizer.decode(output, skip_special_tokens=True) 
                       for output in outputs]
        return translations
```

## 5. 问答系统

### 5.1 检索式问答

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class RetrievalQA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.qa_pairs = []
        self.qa_vectors = None
    
    def add_qa_pair(self, question, answer):
        """添加问答对"""
        self.qa_pairs.append((question, answer))
    
    def build_index(self):
        """构建索引"""
        questions = [qa[0] for qa in self.qa_pairs]
        self.qa_vectors = self.vectorizer.fit_transform(questions)
    
    def answer(self, question, top_k=3):
        """回答问题"""
        if self.qa_vectors is None:
            self.build_index()
        
        # 计算问题相似度
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.qa_vectors).flatten()
        
        # 获取最相似的问答对
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        answers = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 相似度阈值
                answers.append({
                    'answer': self.qa_pairs[idx][1],
                    'similarity': similarities[idx],
                    'question': self.qa_pairs[idx][0]
                })
        
        return answers
```

### 5.2 生成式问答

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class GenerativeQA:
    def __init__(self, model_name='deepset/roberta-base-squad2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    def answer(self, question, context):
        """生成答案"""
        inputs = self.tokenizer(
            question, 
            context, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取答案起始和结束位置
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # 解码答案
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )
        
        return answer
```

## 6. 情感分析

### 6.1 基于词典的方法

```python
class LexiconBasedSA:
    def __init__(self):
        self.positive_words = set(['好', '棒', '优秀', '喜欢', '满意', '开心'])
        self.negative_words = set(['差', '坏', '糟糕', '讨厌', '失望', '难过'])
        self.intensifiers = {'非常': 2, '很': 1.5, '比较': 0.8, '有点': 0.5}
        self.negations = {'不', '没', '无', '非'}
    
    def analyze_sentiment(self, text):
        """分析情感"""
        words = text.split()
        score = 0
        negation_count = 0
        
        for i, word in enumerate(words):
            if word in self.negations:
                negation_count += 1
            elif word in self.positive_words:
                word_score = 1
                # 检查强度词
                if i > 0 and words[i-1] in self.intensifiers:
                    word_score *= self.intensifiers[words[i-1]]
                score += word_score * (-1) ** negation_count
                negation_count = 0
            elif word in self.negative_words:
                word_score = -1
                if i > 0 and words[i-1] in self.intensifiers:
                    word_score *= self.intensifiers[words[i-1]]
                score += word_score * (-1) ** negation_count
                negation_count = 0
        
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'
```

### 6.2 基于深度学习的方法

```python
class DeepSA(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super(DeepSA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # 注意力机制
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.classifier(self.dropout(context))
        return output
```

## 总结

NLP核心任务涵盖了自然语言处理的各个方面：

1. **文本分类**：情感分析、主题分类等
2. **命名实体识别**：识别文本中的人名、地名、组织名等
3. **文本摘要**：抽取式和生成式摘要
4. **机器翻译**：统计方法和神经方法
5. **问答系统**：检索式和生成式问答
6. **情感分析**：基于词典和深度学习的方法

每种任务都有其特定的技术栈和评估方法，需要根据具体应用场景选择合适的方法。 

## 中文分词与文本处理详解

# 中文分词与文本处理详解

## 1. 中文分词概述

中文分词（CWS, Chinese Word Segmentation）是NLP的基础任务，目的是将连续的中文字符序列切分为有意义的词语。

**示例**：
- **输入**：`"我爱自然语言处理"`
- **输出**：`["我", "爱", "自然语言处理"]`

## 2. 中文分词的主要技术

### 2.1 基于规则的方法

#### 正向最大匹配（FMM）
```python
def forward_maximum_matching(text, dictionary):
    """正向最大匹配算法"""
    result = []
    i = 0
    while i < len(text):
        matched = False
        for j in range(min(len(text) - i, 10), 0, -1):  # 最大词长10
            word = text[i:i+j]
            if word in dictionary:
                result.append(word)
                i += j
                matched = True
                break
        if not matched:
            result.append(text[i])
            i += 1
    return result

# 示例词典
dictionary = {"我", "爱", "自然", "语言", "处理", "自然语言", "自然语言处理"}
text = "我爱自然语言处理"
result = forward_maximum_matching(text, dictionary)
print(result)  # ['我', '爱', '自然语言处理']
```

#### 双向最大匹配
```python
def bidirectional_matching(text, dictionary):
    """双向最大匹配"""
    forward_result = forward_maximum_matching(text, dictionary)
    backward_result = backward_maximum_matching(text, dictionary)
    
    # 选择词数更少的结果
    if len(forward_result) <= len(backward_result):
        return forward_result
    else:
        return backward_result
```

### 2.2 基于统计机器学习的方法

#### 隐马尔可夫模型（HMM）
将分词视为序列标注问题：
- B：词首（Begin）
- M：词中（Middle）
- E：词尾（End）
- S：单字词（Single）

```python
import numpy as np
from sklearn.metrics import accuracy_score

class HMMSegmenter:
    def __init__(self):
        self.states = ['B', 'M', 'E', 'S']
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probs = None
    
    def train(self, sentences, labels):
        """训练HMM模型"""
        # 计算转移概率
        self.transition_matrix = np.zeros((4, 4))
        self.emission_matrix = {}
        self.initial_probs = np.zeros(4)
        
        for sentence, label_seq in zip(sentences, labels):
            # 统计转移概率
            for i in range(len(label_seq) - 1):
                curr_state = self.states.index(label_seq[i])
                next_state = self.states.index(label_seq[i + 1])
                self.transition_matrix[curr_state][next_state] += 1
            
            # 统计发射概率
            for char, state in zip(sentence, label_seq):
                if char not in self.emission_matrix:
                    self.emission_matrix[char] = np.zeros(4)
                self.emission_matrix[char][self.states.index(state)] += 1
            
            # 初始概率
            self.initial_probs[self.states.index(label_seq[0])] += 1
        
        # 归一化
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        self.initial_probs = self.initial_probs / self.initial_probs.sum()
        
        for char in self.emission_matrix:
            self.emission_matrix[char] = self.emission_matrix[char] / self.emission_matrix[char].sum()
    
    def viterbi_decode(self, sentence):
        """维特比算法解码"""
        n = len(sentence)
        dp = np.zeros((n, 4))
        backpointer = np.zeros((n, 4), dtype=int)
        
        # 初始化
        for i in range(4):
            char = sentence[0]
            if char in self.emission_matrix:
                dp[0][i] = self.initial_probs[i] * self.emission_matrix[char][i]
            else:
                dp[0][i] = self.initial_probs[i] * 1e-10
        
        # 动态规划
        for t in range(1, n):
            for j in range(4):
                max_prob = -np.inf
                best_state = 0
                for i in range(4):
                    prob = dp[t-1][i] * self.transition_matrix[i][j]
                    if prob > max_prob:
                        max_prob = prob
                        best_state = i
                
                char = sentence[t]
                if char in self.emission_matrix:
                    dp[t][j] = max_prob * self.emission_matrix[char][j]
                else:
                    dp[t][j] = max_prob * 1e-10
                backpointer[t][j] = best_state
        
        # 回溯
        best_path = []
        best_state = np.argmax(dp[n-1])
        for t in range(n-1, -1, -1):
            best_path.append(self.states[best_state])
            best_state = backpointer[t][best_state]
        
        return list(reversed(best_path))
```

#### 条件随机场（CRF）
```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, x, mask):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions
    
    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask, reduction='mean')
    
    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)
```

### 2.3 基于深度学习的方法

#### BERT+CRF
```python
from transformers import BertModel, BertTokenizer

class BERT_CRF_Segmenter(nn.Module):
    def __init__(self, bert_model_name, num_tags):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return predictions
```

## 3. 常用分词工具库

### 3.1 Jieba分词
```python
import jieba

# 精确模式（默认）
text = "我爱自然语言处理"
print(jieba.lcut(text))  # ['我', '爱', '自然语言', '处理']

# 全模式（所有可能分词）
print(jieba.lcut(text, cut_all=True))  # ['我', '爱', '自然', '自然语言', '语言', '处理']

# 搜索引擎模式
print(jieba.lcut_for_search(text))  # ['我', '爱', '自然', '语言', '自然语言', '处理']

# 添加自定义词典
jieba.load_userdict("custom_dict.txt")
```

### 3.2 LTP分词
```python
from ltp import LTP

ltp = LTP()
text = "我爱自然语言处理"
seg, _ = ltp.seg([text])
print(seg)  # [['我', '爱', '自然语言处理']]
```

### 3.3 HanLP分词
```python
import hanlp

# 加载分词模型
tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
text = "我爱自然语言处理"
print(tokenizer(text))  # ['我', '爱', '自然语言处理']
```

## 4. OOV（Out-of-Vocabulary）问题处理

### 4.1 OOV问题的根源
- **罕见词**：专业术语（如"量子纠缠"）、领域特定词汇
- **新词**：网络用语（如"绝绝子"）、品牌名（如"ChatGPT"）
- **形态变化**：未登录的复数、时态
- **拼写错误**：如"accommodate" → "acommodate"

### 4.2 解决方案

#### 子词分割（Subword Tokenization）
```python
from tokenizers import ByteLevelBPETokenizer

# 训练BPE分词器
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["text.txt"], vocab_size=5000)

# 编码文本
encoded = tokenizer.encode("量子纠缠")
print(encoded.tokens)  # 输出可能为["量", "子", "纠", "缠"]
```

#### 字符级表示
```python
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.dropout(concatenated)
        return self.fc(output)
```

#### FastText处理OOV
```python
from gensim.models import FastText

# 训练FastText模型
model = FastText(vector_size=100, window=5, min_count=1)
model.build_vocab(corpus_iterable=texts)
model.train(corpus_iterable=texts, total_examples=len(texts), epochs=10)

# 即使"绝绝子"未在训练集中，也能生成向量
vector = model.wv["绝绝子"]  # 通过子词组合得到
```

## 5. 文本数据清洗

### 5.1 基础清洗
```python
import re
from bs4 import BeautifulSoup
import emoji

def clean_text(text):
    """基础文本清洗"""
    # 去除HTML标签
    text = BeautifulSoup(text, "lxml").get_text()
    
    # 去除URL和邮箱
    text = re.sub(r'http\S+|www\S+|@\w+', '[LINK]', text)
    
    # 处理emoji
    text = emoji.replace_emojis(text, replace='')
    
    # 标准化空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 去除特殊符号（保留中文、英文、数字、基本标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:()]', '', text)
    
    return text
```

### 5.2 高级清洗
```python
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def advanced_clean_text(text):
    """高级文本清洗"""
    # 拼写纠正
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(corrected_text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]
    
    return ' '.join(filtered_words)
```

### 5.3 领域特定清洗
```python
def clean_medical_text(text):
    """医疗文本清洗"""
    # 保留医疗术语
    medical_terms = {"EGFR", "mutation", "symptom", "diagnosis", "treatment"}
    
    # 自定义停用词（排除医疗术语）
    custom_stop_words = set(stopwords.words('english')) - medical_terms
    
    # 清洗过程
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in custom_stop_words]
    
    return ' '.join(filtered_words)
```

## 6. 评估指标

### 6.1 分词评估
```python
def evaluate_segmentation(predicted, gold_standard):
    """评估分词结果"""
    # 计算精确率、召回率、F1值
    pred_words = set(predicted)
    gold_words = set(gold_standard)
    
    precision = len(pred_words & gold_words) / len(pred_words) if pred_words else 0
    recall = len(pred_words & gold_words) / len(gold_words) if gold_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### 6.2 工具库性能对比
| 工具库 | 技术方案 | F1值 | 特点 |
|--------|----------|------|------|
| **Jieba** | 前缀词典 + HMM | ~0.85 | 轻量级，适合通用文本 |
| **SnowNLP** | 统计模型（CRF） | ~0.87 | 适合社交媒体文本 |
| **THULAC** | 结构化感知机 + 深度学习 | ~0.92 | 高准确率，支持多任务 |
| **LTP** | 基于BERT的联合模型 | ~0.97 | 哈工大出品，工业级精度 |
| **HanLP** | 多种模型（CRF/BERT等） | ~0.95 | 支持多语言、多任务 |

## 7. 实际应用示例

### 7.1 完整分词Pipeline
```python
class ChineseTextProcessor:
    def __init__(self, segmenter='jieba'):
        self.segmenter = segmenter
        if segmenter == 'jieba':
            import jieba
            self.tokenizer = jieba
        elif segmenter == 'ltp':
            from ltp import LTP
            self.tokenizer = LTP()
    
    def segment(self, text):
        """分词"""
        if self.segmenter == 'jieba':
            return self.tokenizer.lcut(text)
        elif self.segmenter == 'ltp':
            seg, _ = self.tokenizer.seg([text])
            return seg[0]
    
    def clean_and_segment(self, text):
        """清洗并分词"""
        # 清洗文本
        cleaned_text = clean_text(text)
        
        # 分词
        words = self.segment(cleaned_text)
        
        return words

# 使用示例
processor = ChineseTextProcessor('jieba')
text = "我爱自然语言处理，这是一个很有趣的领域！"
result = processor.clean_and_segment(text)
print(result)  # ['我', '爱', '自然语言处理', '这', '是', '一个', '很', '有趣', '的', '领域']
```

## 8. 关键要点总结

1. **分词方法选择**：根据应用场景选择合适的算法（规则、统计、深度学习）
2. **OOV处理**：使用子词分割、字符级表示等方法处理未登录词
3. **数据清洗**：根据领域特点进行针对性的文本预处理
4. **评估指标**：使用精确率、召回率、F1值评估分词质量
5. **工具选择**：Jieba适合通用场景，LTP适合高精度要求，HanLP适合多任务 

