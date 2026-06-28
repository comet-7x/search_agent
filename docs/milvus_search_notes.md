# Milvus 搜索流程笔记

> 个人学习笔记，持续补充中。

---

## 目录

1. [核心概念速览](#1-核心概念速览)
2. [写入阶段](#2-写入阶段)
   - 2.1 稠密向量（Dense Vector）
   - 2.2 稀疏向量与 BM25
   - 2.3 索引类型
3. [搜索阶段](#3-搜索阶段)
   - 3.1 ANN Search
   - 3.2 全文检索（BM25 Search）
   - 3.3 Hybrid Search
4. [Ranker：多路结果融合](#4-ranker多路结果融合)
   - 4.1 RRFRanker
   - 4.2 WeightedRanker
5. [标量过滤（Filter）](#5-标量过滤filter)
6. [其他重要概念](#6-其他重要概念)
7. [完整 RAG 管道：召回 → 精排 → 生成](#7-完整-rag-管道召回--精排--生成)
   - 7.1 为什么需要两层排序
   - 7.2 Bi-Encoder vs Cross-Encoder
   - 7.3 精排层（Reranker）
   - 7.4 生成层（LLM）
   - 7.5 完整流程总图
8. [实践选择建议](#8-实践选择建议)

---

## 1. 核心概念速览

| 概念 | 一句话 | 所处阶段 |
|------|--------|----------|
| Dense Vector | 每个维度都有值的高维浮点数组，表达语义 | 写入 / 搜索 |
| Sparse Vector | 大多数维度为 0，只有出现过的词有权重 | 写入 / 搜索 |
| BM25 | 基于词频+逆文档频率的经典关键词匹配算法 | 写入（自动） / 搜索 |
| HNSW | 稠密向量的分层图索引，用于快速 ANN | 建索引 |
| SPARSE_INVERTED_INDEX | 稀疏向量的倒排索引，对应搜索引擎原理 | 建索引 |
| ANN Search | 近似最近邻搜索，速度换精度 | 搜索 |
| Hybrid Search | 多路 ANN 并行召回 + Ranker 重排序 | 搜索 |
| AnnSearchRequest | Hybrid Search 中单路搜索的配置封装 | 搜索 |
| RRFRanker | 倒数排名融合，只看排名不看原始分数 | 重排序 |
| WeightedRanker | 加权融合，需手动指定各路权重 | 重排序 |
| Filter | 标量字段的布尔过滤，叠加在向量搜索上 | 搜索 |
| Partition | 集合内的物理数据分片，缩小搜索范围 | 管理 / 搜索 |

---

## 2. 写入阶段

### 2.1 稠密向量（Dense Vector）

```
原始文本: "如何训练大模型"
    │
    ▼  Embedding Model（OpenAI / BGE / text2vec 等）
    │
    ▼
[0.12, -0.45, 0.89, ..., 0.33]   ← 1536 维（维度由模型决定）
    │
    ▼
存入 DataType.FLOAT_VECTOR 字段
```

**关键点**
- 语义相近的句子，在高维空间中距离也近（即使用词完全不同）
- 维度越高，表达能力越强，但存储和计算成本越高
- 每次 insert 都需要自己先调用 Embedding Model 算好再存入

---

### 2.2 稀疏向量与 BM25

```
原始文本: "如何训练大模型"
    │
    ▼  Milvus 服务端 BM25 Function（在 Schema 里配置，自动触发）
    │
    ▼
{词ID_42: 0.87, 词ID_156: 0.43, 词ID_789: 1.20, ...}
← 只有文档中出现过的词才有非零值（稀疏）
    │
    ▼
存入 DataType.SPARSE_FLOAT_VECTOR 字段（自动，不需要手动传值）
```

#### BM25 公式原理

$$\text{score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t,D) \cdot (k_1 + 1)}{f(t,D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

| 参数 | 含义 |
|------|------|
| `f(t, D)` | 词 t 在文档 D 中的出现次数（词频 TF） |
| `IDF(t)` | 词 t 的逆文档频率，越罕见值越高 |
| `k1` | 词频饱和参数，防止高频词权重无限增大（通常 1.2~2.0） |
| `b` | 文档长度归一化因子（通常 0.75） |
| `avgdl` | 语料库平均文档长度 |

**BM25 vs 稠密向量对比**

| | 稠密向量 ANN | BM25 全文检索 |
|--|--|--|
| 查"苹果手机" | 能找到"iPhone"、"智能设备" | 只匹配含"苹果"或"手机"的文档 |
| 查"GPT-4o" | 可能被语义噪声干扰 | 精确匹配专有名词 |
| 优势 | 语义理解、同义词泛化 | 关键词精确、可解释性强 |
| 劣势 | 专有名词/新词效果差 | 无法理解语义，不能泛化 |

#### Schema 配置示例（BM25 自动生成稀疏向量）

```python
# text 字段：存原始文本，启用分析器
text: str = Field(..., json_schema_extra={
    "datatype": DataType.VARCHAR,
    "max_length": 65535,
    "enable_analyzer": True,
})

# sparse 字段：由服务端 BM25 Function 自动生成，insert 时不传此字段
sparse: dict | None = Field(default=None, json_schema_extra={
    "datatype": DataType.SPARSE_FLOAT_VECTOR,
    "function": {
        "name": "sparse_bm25",
        "input_field_names": ["text"],   # 从 text 字段生成
        "function_type": FunctionType.BM25,
    },
    "index": {
        "metric_type": "BM25",
        "index_type": "SPARSE_INVERTED_INDEX",
        "index_name": "sparse_index",
        "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
    },
})
```

---

### 2.3 索引类型

#### HNSW（Hierarchical Navigable Small World）— 稠密向量

分层图结构，高层节点少（远距离跳转），低层节点密集（精确邻居）。

```
Layer 2:  A ─────────────── H          ← 少量节点，快速跨越大距离
Layer 1:  A ─── C ─── E ─── H
Layer 0:  A-B-C-D-E-F-G-H-I-J         ← 所有节点，精确邻居连接
```

| 参数 | 作用 | 影响 |
|------|------|------|
| `M` | 每个节点最大邻居连接数 | 越大精度越高，内存越多 |
| `efConstruction` | 建索引时的候选探索范围 | 越大索引质量越好，建索引越慢 |
| `ef`（搜索时） | 搜索时维护的候选集大小 | 越大召回率越高，必须 ≥ limit |
| `metric_type` | 距离度量方式 | COSINE / L2 / IP |

> **COSINE vs L2 vs IP**
> - COSINE：余弦相似度，关注方向不关注长度，文本场景首选
> - L2：欧式距离，关注绝对位置
> - IP：内积，向量已归一化时等价于 COSINE

#### SPARSE_INVERTED_INDEX — 稀疏向量

倒排索引：`词 → [文档ID列表]`，与搜索引擎原理一致，天然适配 BM25。

| 参数 | 说明 |
|------|------|
| `metric_type: "BM25"` | 固定写法，专用于 BM25 稀疏向量 |
| `inverted_index_algo` | `DAAT_MAXSCORE`（文档级评分，推荐）|

#### 其他索引（了解）

| 索引 | 适用 | 特点 |
|------|------|------|
| IVF_FLAT | 稠密向量 | 聚类+全量扫描，精度高，速度中 |
| IVF_SQ8 | 稠密向量 | 量化压缩，内存小，精度略降 |
| IVF_PQ | 稠密向量 | 乘积量化，压缩率最高 |
| AUTOINDEX | 稠密向量 | Milvus 自动选择，云托管推荐 |

---

## 3. 搜索阶段

### 3.1 ANN Search（近似最近邻搜索）

**ANN = Approximate Nearest Neighbor**，核心取舍：**速度换精度**。

精确搜索需 O(n) 与每条数据比较，百万级不可接受。ANN 通过索引在毫秒级找到"足够近"的结果，召回率通常 > 95%。

```
查询向量 [0.1, 0.2, ...]
    │
    ▼  从 HNSW 顶层快速定位大致区域
    │
    ▼  逐层下钻，维护大小为 ef 的候选集
    │
    ▼  对候选集计算精确距离
    │
    ▼  按 distance 排序，返回 Top-K
```

```python
results = await storage.search(
    collection="my_col",
    data=[[0.1, 0.2, ...]],          # 查询向量
    anns_field="embedding",           # 指定向量字段
    limit=10,
    search_params={
        "metric_type": "COSINE",
        "params": {"ef": 64},         # 候选集大小，≥ limit
    },
    output_fields=["text", "source"],
)
```

---

### 3.2 全文检索（BM25 Search）

本质也是通过 `search` 接口，传文本而非向量，指向稀疏向量字段：

```
查询: "大模型训练技巧"
    │
    ▼  Milvus 将查询文本转为 BM25 查询稀疏向量
    │
    ▼  SPARSE_INVERTED_INDEX 查找包含这些词的文档
    │
    ▼  BM25 公式计算每个文档的相关性分数
    │
    ▼  按分数排序，返回 Top-K
```

```python
results = await storage.search(
    collection="my_col",
    data=["大模型训练技巧"],           # 传文本字符串
    anns_field="sparse",              # 指向稀疏向量字段
    limit=10,
    search_params={"metric_type": "BM25"},
    output_fields=["text"],
)
```

---

### 3.3 Hybrid Search（混合搜索）

**目的**：语义向量搜索 + 关键词全文检索互补，覆盖两者的盲区。

```
查询文本
  │
  ├──────────────────────────────────────┐
  │                                      │
  ▼                                      ▼
AnnSearchRequest #1               AnnSearchRequest #2
稠密向量 ANN                       BM25 稀疏向量
  │                                      │
  ▼                                      ▼
召回 Top-100（语义相关）          召回 Top-100（关键词匹配）
  │                                      │
  └──────────────┬───────────────────────┘
                 ▼
           Ranker 重排序
                 │
                 ▼
           最终 Top-K 结果
```

```python
from pymilvus import AnnSearchRequest, RRFRanker

dense_req = AnnSearchRequest(
    data=[[0.1, 0.2, ...]],           # 查询的稠密向量
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=100,                         # 单路召回数，通常远大于最终 limit
)

sparse_req = AnnSearchRequest(
    data=["大模型训练技巧"],            # 查询文本（BM25 自动转换）
    anns_field="sparse",
    param={"metric_type": "BM25"},
    limit=100,
)

results = await storage.hybrid_search(
    collection="my_col",
    requests=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),
    limit=10,                          # 最终返回条数
    output_fields=["text", "source"],
)
```

---

## 4. Ranker：多路结果融合

### 4.1 RRFRanker（Reciprocal Rank Fusion，倒数排名融合）

$$\text{score}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}$$

| 变量 | 含义 |
|------|------|
| `k` | 平滑参数，默认 60，防止排名极靠前的结果权重过大 |
| `rank_i(d)` | 文档 d 在第 i 路搜索中的排名（从 1 开始） |

**示例计算（k=60）**

```
doc_A: ANN 排第1, BM25 排第2
  → 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325

doc_B: ANN 排第2, BM25 排第1
  → 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
```

**核心优势**
- 无需归一化（ANN 的 cosine 分数 vs BM25 的 TF-IDF 分数量纲完全不同，RRF 只看排名）
- 对两路都排名靠前的文档给予显著奖励
- 参数少，鲁棒性强，是 Hybrid Search 的默认首选

---

### 4.2 WeightedRanker（加权融合）

$$\text{score}(d) = w_1 \cdot \hat{s}_1(d) + w_2 \cdot \hat{s}_2(d)$$

各路分数先归一化到 [0,1]，再按权重加权求和。

```python
WeightedRanker(0.7, 0.3)   # 稠密向量权重 70%，BM25 权重 30%
```

**适用场景**：明确知道哪路信号更重要时使用，例如：
- 语义理解为主：`WeightedRanker(0.8, 0.2)`
- 关键词匹配为主（法律/医疗）：`WeightedRanker(0.3, 0.7)`

**RRF vs Weighted 选择**

| | RRFRanker | WeightedRanker |
|--|--|--|
| 是否需要调参 | 几乎不需要（k=60 通常够用） | 需要手动调权重 |
| 对分数量纲要求 | 无要求 | 需要归一化 |
| 适用场景 | 通用场景、冷启动 | 已知各路重要性时 |

---

## 5. 标量过滤（Filter）

在向量搜索基础上叠加结构化字段过滤：

```python
results = await storage.search(
    collection="my_col",
    data=[[0.1, 0.2, ...]],
    filter="category == 'tech' and publish_date > '2024-01-01'",
    limit=10,
)
```

**两种过滤执行模式**（Milvus 自动选择）

| 模式 | 流程 | 适用场景 |
|------|------|----------|
| Pre-filter | 先过滤标量 → 再 ANN | 过滤率高（大量数据被过滤掉）|
| Post-filter | 先 ANN → 再过滤 | 过滤率低，索引利用率高 |

> 过滤率过高时 Pre-filter 可能导致 ANN 候选集过小，返回结果不足 limit。

---

## 6. 其他重要概念

### Partition（分区）

集合内的物理数据分片，搜索时通过 `partition_names` 限定范围，减少扫描数据量。

```python
# 按业务隔离数据，搜索时只查指定分区
results = await storage.search(
    collection="my_col",
    data=[[...]],
    partition_names=["partition_2024", "partition_2025"],
    limit=10,
)
```

**适用场景**：多租户隔离、按时间分片、按业务类型分片。

---

### Load / Release

Milvus 集合必须 **load 到内存** 后才能搜索，release 后从内存卸载。

```python
await storage.load_collection("my_col")   # 加载到内存
# ... 执行搜索 ...
await storage.release_collection("my_col") # 释放内存
```

**注意**：`create_collection` 时 Milvus 会自动 load，正常使用不需要手动调用。

---

### search_iterator / query_iterator（同步专用）

大量数据分批导出时使用，避免一次性拉取全部数据导致内存爆炸。

```python
# 同步版本（MilvusStorage）
iterator = storage.search_iterator(
    collection="my_col",
    data=[[0.1, 0.2, ...]],
    batch_size=1000,
    anns_field="embedding",
)
# 迭代获取结果
while True:
    batch = iterator.next()
    if not batch:
        iterator.close()
        break
    process(batch)
```

> AsyncMilvusClient 目前不支持 search_iterator，只有同步 MilvusClient 支持。

---

### 一致性级别（Consistency Level）

控制写入后多久能被搜索到：

| 级别 | 含义 | 性能 |
|------|------|------|
| Strong | 写入立即可见 | 最慢 |
| Bounded | 有限延迟后可见（默认）| 中 |
| Session | 同一 session 写入立即可见 | 中 |
| Eventually | 最终一致，可能有较长延迟 | 最快 |

---

## 7. 完整 RAG 管道：召回 → 精排 → 生成

### 7.1 为什么需要两层排序

```
全量数据（百万条）
    │
    │  第一层：召回（Retrieval）
    │  目标：高召回率，宁可多捞，不能漏
    │  工具：ANN + BM25 + RRF  ← Milvus 内部完成
    │  速度：极快（毫秒级）
    ▼
候选集（Top-100 ~ 200 条）
    │
    │  第二层：精排（Reranking）
    │  目标：高精确率，从候选里找出真正最相关的
    │  工具：Cross-Encoder Reranker  ← 应用层完成
    │  速度：较慢（百毫秒级，但候选集已很小）
    ▼
精排结果（Top-5 ~ 20 条）
    │
    │  第三层：生成（Generation）
    │  工具：LLM
    ▼
最终回答
```

Cross-Encoder 太慢，无法直接处理百万数据；ANN 太粗，排名不够准。两层设计让各自发挥优势。

---

### 7.2 Bi-Encoder vs Cross-Encoder

这是理解为什么精排更准的核心：

```
【Bi-Encoder — 召回层使用的 Embedding Model】

Query ──► Encoder ──► 向量 q ─╮
                               ╰─ cosine ──► 分数
Doc   ──► Encoder ──► 向量 d ─╯

两者独立编码，文档向量可提前算好并存入 Milvus
缺点：看不到 query 词和 doc 词之间的细粒度交互
```

```
【Cross-Encoder — 精排层使用的 Reranker】

"[Query] 如何优化 Transformer [SEP] [Doc] 注意力机制..."
        └────────────────────────────────────┘
                拼接后一起送入 Encoder
                        │
                        ▼
                   相关性分数（0~1）

全局注意力同时覆盖 query 和 doc 的每个词对
优点：能精确捕捉词级别的语义匹配
缺点：每次查询都要重新计算，不能缓存
```

| | Bi-Encoder | Cross-Encoder |
|--|--|--|
| 编码方式 | query 和 doc 分开 | (query, doc) 拼接一起 |
| 文档向量可缓存 | ✅ 提前算好存 Milvus | ❌ 依赖 query，无法预算 |
| 查询速度 | 极快（毫秒）| 较慢（需逐对计算）|
| 精度 | 中 | 高 |
| 适用规模 | 百万条数据 | 百条候选 |
| 所处阶段 | 召回层 | 精排层 |

---

### 7.3 精排层（Reranker）

#### 常用模型

| 模型 | 提供方 | 部署方式 | 擅长场景 |
|------|--------|----------|----------|
| BGE-Reranker-v2-m3 | BAAI | 本地开源 | 中英文通用，性价比高 |
| Cohere Rerank | Cohere | API | 商业精度高 |
| Jina Reranker v2 | Jina AI | 本地 / API | 多语言、长文档 |
| Voyage Rerank | Voyage AI | API | 代码和技术文档 |

#### 精排代码示意

```python
# ① 召回阶段：Milvus hybrid_search，故意多召回
candidates = await storage.hybrid_search(
    collection="my_col",
    requests=[dense_req, sparse_req],
    ranker=RRFRanker(),
    limit=100,                        # 多召回，为精排提供足够候选
    output_fields=["text", "doc_id"],
)

# ② 精排阶段：Cross-Encoder 逐对打分
query = "Transformer 注意力机制怎么优化？"
pairs = [(query, c.record["text"]) for c in candidates]

scores = reranker.compute_score(pairs)  # 以 BGE-Reranker 为例

# ③ 按新分数重排，取 Top-10
reranked = sorted(
    zip(scores, candidates),
    key=lambda x: x[0],
    reverse=True,
)[:10]
```

#### RRF/Weighted 与 Reranker 的本质区别

```
召回层内部的 RRF / WeightedRanker：
  · 目的：把多路 ANN 结果列表合并为一个排名
  · 方法：基于排名位置（RRF）或归一化分数（Weighted）
  · 不理解 query 和 doc 的文本内容
  · 在 Milvus 内部，微秒级

精排层的 Cross-Encoder Reranker：
  · 目的：重新评估候选与 query 的真实语义相关性
  · 方法：(query, doc) 联合输入，全注意力交互
  · 真正"读懂"了 query 和 doc 的词级语义匹配
  · 在应用层，百毫秒级
```

---

### 7.4 生成层（LLM）

```python
context = "\n\n".join(
    f"[{i+1}] {r[1].record['text']}"
    for i, r in enumerate(reranked)
)

prompt = f"""你是一个专业助手，请根据以下参考资料回答用户问题。
如果资料中没有相关信息，请如实说明。

参考资料：
{context}

用户问题：{query}"""

answer = llm.chat(prompt)
```

**上下文排列技巧（Lost in the Middle）**：研究表明 LLM 对 prompt 开头和结尾的内容注意力更强，中间部分容易被忽略。可以把最相关的文档放在首位和末位。

---

### 7.5 完整流程总图

```
用户输入："Transformer 注意力机制怎么优化？"
         │
         │ ① 查询预处理（可选）
         │   · Query Rewriting：改写成更适合检索的形式
         │   · HyDE：让 LLM 先生成假设答案，用答案向量检索
         ▼
══════════════════ 召回层（粗筛）══════════════════
         │
         ├─► Embedding Model → 稠密向量
         │       └─► Milvus ANN Search (HNSW)      → Top-100
         │
         ├─► BM25 文本分析
         │       └─► Milvus 全文检索                → Top-100
         │
         └─► Hybrid Search + RRFRanker 融合         → Top-100
                                                    （候选集）
══════════════════ 精排层（精选）══════════════════
         │
         ▼
   Cross-Encoder Reranker
   逐一计算 (query, doc_i) 相关性分数
         │
         ▼
         Top-10（精排后，真正最相关）
══════════════════ 生成层（利用）══════════════════
         │
         ▼
   Prompt = System Prompt + Context(Top-10) + Query
         │
         ▼
         LLM
         │
         ▼
   最终回答
```

---

## 旧版流程总图（仅 Milvus 内部）

```
【写入阶段】

原始文本 "如何优化 Transformer 注意力机制"
  │
  ├─► 调用 Embedding Model
  │       └─► [0.12, -0.45, ..., 0.33]  →  FLOAT_VECTOR 字段（embedding）
  │
  └─► 只需传 text 字段，Milvus 自动
          └─► {Transformer:0.9, 注意力:0.7, ...}  →  SPARSE_FLOAT_VECTOR 字段（sparse）

                    ↓ insert

           Milvus Collection
           ┌──────────────────────────────────────┐
           │ id │ text │ embedding │ sparse │ ... │
           └──────────────────────────────────────┘
                       ↓ 建索引
           ┌──────────────┬───────────────────────┐
           │  HNSW 索引   │  SPARSE_INVERTED_INDEX │
           │（稠密向量）   │  （BM25 稀疏向量）      │
           └──────────────┴───────────────────────┘


【搜索阶段】

用户查询: "Transformer 注意力优化"
  │
  ├─► 稠密向量化 → [0.08, -0.41, ...]
  │       └─► AnnSearchRequest #1（HNSW ANN） → 召回 Top-100
  │
  ├─► BM25 文本 → {Transformer:0.8, 注意力:0.6, ...}
  │       └─► AnnSearchRequest #2（SPARSE_INVERTED_INDEX） → 召回 Top-100
  │
  └─► hybrid_search
          │
          ▼
      RRFRanker(k=60)：只看排名，融合两路结果
          │
          ▼
      filter: "category == 'paper'"（可选）
          │
          ▼
      最终 Top-10 结果
      [VectorSearchResult(record={text:..., id:...}, score=0.0327), ...]
```

---

## 8. 实践选择建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 通用 RAG / 问答 | Hybrid Search + RRFRanker | 覆盖语义和关键词，无需调参 |
| 纯语义搜索（推荐/相似内容） | 单路 ANN Search | 简单高效，无需 BM25 |
| 法律/医疗/代码（术语精确匹配重要）| Hybrid Search + WeightedRanker(0.3, 0.7) | 提高关键词权重 |
| 按关键词精确查找 | 单路 BM25 Search | 无需 Embedding，低成本 |
| 大量数据导出 | query_iterator / search_iterator | 分批避免内存溢出（仅同步） |
| 多租户 / 按时间分片 | 配合 Partition | 减少搜索范围，提升性能 |

---

## 待补充

- [ ] Milvus 向量量化（INT8_VECTOR / BINARY_VECTOR）在移动/边缘场景的应用
- [ ] GPU 索引（GPU_IVF_FLAT / GPU_CAGRA）与 CPU 索引对比
- [ ] ~~Reranker（Cross-Encoder）在 Milvus 返回结果后的二次精排~~ ✅ 已补充至第 7 节
- [ ] 多向量字段场景下 Hybrid Search 超过两路的设计
- [ ] `search_params.ef` 与 `limit` 的调优经验
- [ ] Milvus Cluster 模式 vs Standalone 模式的数据分布差异
- [ ] Lost in the Middle 问题与上下文排列策略
- [ ] Query Rewriting 和 HyDE 的具体实现
- [ ] 召回数量（Top-K）与最终输出数量（Top-N）的调优经验
