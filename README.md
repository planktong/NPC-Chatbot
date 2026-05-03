# NPC-Chatbot
Agentic Hybrid RAG LLM and EHR Visual

## RAG pipeline 的逻辑
主逻辑在 `backend/rag_pipeline.py`，它用 **LangGraph 的 StateGraph** 把检索、相关性评估、必要时的查询改写/扩展串成一个小工作流。代码入口是 `run_rag_graph(question)`，最终产出 `docs/context/rag_trace` 给上层 Agent 去生成回答。  
（文件：`backend/rag_pipeline.py` citeturn0commentary0）

---

## 1) 整体流程（LangGraph 状态机）

图的结构（`build_rag_graph()`）是：

1. **retrieve_initial**：初次检索
2. **grade_documents**：用 LLM 判断“检索到的上下文是否相关”
3. 条件分支：
   - 相关（yes）→ **END**（结束：直接用当前 context 生成回答）
   - 不相关/无法打分 → **rewrite_question**（查询扩展/重写）
4. **retrieve_expanded**：用扩展查询再次检索 → **END**

对应代码在 `build_rag_graph()`：`retrieve_initial -> grade_documents -> (generate_answer|rewrite_question)`，`rewrite_question -> retrieve_expanded -> END`。 citeturn0commentary0

---

## 2) retrieve_initial：初次检索（核心是 rag_utils.retrieve_documents）

`retrieve_initial(state)` 做的事：

- 取 `query = state["question"]`
- 调 `retrieve_documents(query, top_k=15, entity_query=question)`
- 从返回里取：
  - `docs`：检索到的 chunk 列表
  - `merged_context`：已经合并好的上下文（可能包含图谱 + 向量片段拼接）
  - `meta`：检索元信息（候选数、leaf 层级、是否 auto-merge、是否 rerank、graph 是否启用等）
- 生成 `context`：
  - 优先用 `merged_context`
  - 否则用 `_format_docs(docs)` 把 chunk 拼成文本
- 组装 `rag_trace`（给前端/日志展示的可解释字段）

这一步还会通过 `emit_rag_step(...)` 往前端“思考气泡”推送进度信息。 citeturn0commentary0

---

## 3) grade_documents：相关性打分（决定是否要重写查询）

`grade_documents_node(state)`：

- 用 `_get_grader_model()` 初始化一个 grader 模型（环境变量 `GRADE_MODEL`，默认 `"gpt-4.1"`）。
- 把 `question + context` 填进 `GRADE_PROMPT`，让模型输出结构化字段 `binary_score: yes/no`
- 如果 `yes`：route = `"generate_answer"`（图直接 END）
- 如果 `no`：route = `"rewrite_question"`（进入查询扩展）
- 如果 grader 模型不可用：默认走 rewrite_question（score unknown）

这一步只负责“路由选择”，不生成最终回答。 citeturn0commentary0

---

## 4) rewrite_question：选择扩展策略（step_back / hyde / complex）

`rewrite_question_node(state)`：

- 用 `_get_router_model()`（一般是 `MODEL`）来选择策略：
  - `step_back`：生成“退步问题 + 退步答案”，拼成 expanded_query（用 `step_back_expand`）
  - `hyde`：生成一段“假设性文档”作为检索 query（`generate_hypothetical_document`）
  - `complex`：两者都做（HyDE + step_back）
- 把 strategy、expanded_query 等信息写回 state，并同步到 `rag_trace` 的 `rewrite_strategy/rewrite_query`。 citeturn0commentary0

---

## 5) retrieve_expanded：扩展检索（可能多路召回再合并）

`retrieve_expanded(state)` 的特点是：**按策略可能做 1 路或 2 路检索，然后去重合并**：

- 如果策略包含 `hyde`：
  - 用 hypothetical_doc 做 `retrieve_documents(hypothetical_doc, ...)`
  - 注意：`graph_on_hyde = (strategy == "hyde")`  
    - 纯 hyde：允许 include_graph_merge
    - complex：HyDE 这路默认不合并图谱（图谱只在 step_back 那路合并一次）
- 如果策略包含 `step_back`：
  - 用 expanded_query 做 `retrieve_documents(expanded_query, include_graph_merge=True, ...)`
- 把两路结果 `results.extend(...)` 合并后：
  - 按 (filename, page_number, text) 去重
  - 重新编号 `rrf_rank`（只是展示用，避免两路各自 rank 重复）
  - context 拼接：`hyde_merged_context + "========" + step_merged_context`

最后把 expanded 阶段的 docs/context 写入 state，并把一堆统计（rerank、auto_merge、dense/sparse count、graph fields）汇总进 `rag_trace`。 citeturn0commentary0

---

## 6) 这个 pipeline 的“检索引擎”到底做了什么？

`rag_pipeline.py` 自己不直接连 Milvus/Neo4j，它把检索全部委托给 `rag_utils.retrieve_documents(...)`。在你仓库里该函数会做（简述）：

- 对 query 生成 dense + sparse embedding
- **Milvus dense 检索** + **Milvus sparse(BM25) 检索**
- 用本地 RRF (`_compute_rrf`) 融合两路候选
- 可选 rerank（看环境变量是否配置）
- **Auto-merging**：命中足够多子块时，用父块替换（父块来自 `ParentChunkStore`）
- 可选把 **Neo4j 图谱上下文** 合并到最终 merged_context

（这些细节在 `backend/rag_utils.py`，但你的问题问的是 pipeline，所以我这里只点到为止。）

---
## graph在这里的作用是什么
这里的 **graph（Neo4j 知识图谱）**作用是：在向量检索（Milvus dense/sparse）之外，额外提供一份**“实体关系驱动”的结构化背景上下文**，并把它**合并进最终给 LLM 的 prompt context**，用来提升回答的完整性/可解释性，尤其是在“问法包含明确疾病/药物/基因等实体”或“向量召回不足”的场景。

结合你仓库的实现，graph 的具体作用点主要有 3 个：

---

## 1) 作为“向量检索的补充上下文”，合并进 merged_context

`rag_utils.retrieve_documents()` 在拿到向量召回的 chunks 后，会调用 `_merge_graph_and_vector_context(...)`：

- 用 `MedicalGraphRAGRetriever.build_trace_for_ui(entity_query, top_k=5)` 生成 `graph_text`（图谱上下文文本）和 `subgraph/entities` 等结构信息
- 再把 `graph_text` 与向量 chunks 格式化后的文本拼在一起，形成 `merged_context`
- 这份 `merged_context` 会被 `rag_pipeline.retrieve_initial` / `retrieve_expanded` 优先用作 `state["context"]` 给后续 LLM 生成回答使用。 citeturn0commentary0

也就是说：**graph 不直接替代向量检索，而是把“关系/事实链”塞进 prompt，帮助模型补齐关联信息。**

---

## 2) 在“向量无命中”时提供兜底（graph_only）

如果向量侧 RRF 融合后 `retrieved` 为空，`retrieve_documents()` 会走一个分支：

- 仍然尝试构建 `merged_context`（此时向量 chunks 为空列表）
- 如果图谱返回了内容，就会变成 **graph_only** 的检索模式（代码里会记录 `retrieval_mode` 为 `graph_only` 或类似含义，并通过 `emit_rag_step` 提示）。 citeturn0commentary0

所以 graph 还有一个很实用的作用：**当 Milvus 召回失败/太弱时，让系统不至于完全没上下文。**

---

## 3) 提供可解释信息：实��、子图、预览（给前端/调试）

在 `rag_pipeline.retrieve_initial` 组装的 `rag_trace` 里，会记录这些 graph 相关字段：

- `graph_kb_applied`：是否确实用了图谱上下文
- `graph_entities`：从 query 里识别到的实体（疾病/药物/基因…）
- `graph_subgraph`：子图 nodes/edges（用于可视化或调试）
- `graph_context_preview`：图谱上下文的预览片段
- `graph_error`：异常信息（如果有）

这些主要用于 **UI 展示“我到底查到了什么实体关系”**、以及开发调试。 citeturn0commentary0

---

## 在 rag_pipeline 里，graph 什么时候会被用？

graph 的开关不是在 pipeline 节点里直接控制，而是通过调用 `retrieve_documents(..., include_graph_merge=...)` 传进去的：

- `retrieve_initial(...)`：默认走 `retrieve_documents(..., include_graph_merge=True)`（参数未显式传时，函数默认值是 True）
- `retrieve_expanded(...)`：
  - **step_back**：`include_graph_merge=True`
  - **hyde**：只有在纯 hyde 策略时才 `include_graph_merge=True`；如果是 **complex**，HyDE 那路通常不合并图谱（避免重复），图谱只在 step_back 那路合并一次。 citeturn0commentary0

---

##  `retrieve_initial(state)` 的执行顺序
你可以把它理解成：**用户问一句话 → 系统去知识库找资料 → 整理成一段“可喂给大模型的参考材料”**。

> 对应源码在：`backend/rag_pipeline.py` 的 `retrieve_initial` 函数。citeturn0commentary0

---

## 0) `state` 是什么？
`state` 可以理解成这个流水线随身携带的“工作记录本/表单”。里面至少有：

- `state["question"]`：用户原始问题（例如“糖尿病怎么用药？”）
- 后面会逐步往里面填：
  - `docs`：找到了哪些资料片段
  - `context`：最后整理出的参考文本（给大模型用）
  - `rag_trace`：过程日志（给前端展示“我做了哪些检索步骤”）

---

## 1) 取出 query = 用户问题
```python
query = state["question"]
```
含义：**把用户的提问当作检索关键词/检索语句**。

- 你可以把它类比成：你在图书馆检索系统里输入的那句话。
- 为什么先这么做：最直接、最快，先试一次“原问题检索”。

---

## 2) `emit_rag_step`：往前端显示“我正在检索”
```python
emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
```
这不是检索本身，而是**进度提示**。

- 前端会显示类似：“正在检索知识库… 查询：xxxx”
- 目的：让用户看到系统在工作，不是卡住。

---

## 3) 调用 `retrieve_documents(...)`：真正开始“找资料”
```python
retrieved = retrieve_documents(query, top_k=15, entity_query=state["question"])
```

这一步相当于：**去两个“检索引擎”找资料，然后融合、重排、合并**。虽然你没学过代码，但可以按“功能模块”理解它在做什么：

### 3.1 它会去哪里找？
主要是两类来源（在你的项目里）：

1) **Milvus 向量库**：像“语义搜索”，不是按字面匹配，而是按意思相近找段落  
2) **稀疏/BM25**：更像传统“关键词搜索”（比如包含某个词就容易被召回）

另外还可能有：
- **Neo4j 图谱（graph）**：把图谱里跟实体相关的关系/知识拼成额外背景（如果启用）

### 3.2 top_k=15 是什么意思？
可以理解为：**最多先拿 15 个候选片段**（候选多一点，后面再筛、再合并更稳）。

### 3.3 entity_query 为什么还传一次 question？
图谱检索通常要先识别实体（疾病/药物/基因等），所以用原始问��来做实体匹配更合适。

---

## 4) 从返回结果里取三个东西：docs / meta / merged_context
```python
results = retrieved.get("docs", [])
retrieve_meta = retrieved.get("meta", {})
merged = (retrieved.get("merged_context") or "").strip()
```

把它们理解成：

### 4.1 `docs`（这里叫 results）
**一组“找到的资料片段”**（每个片段大概包含：来自哪个文件、哪一页、文本内容、评分等）。

- 类比：你在搜索结果里看到的多条“段落摘录”。

### 4.2 `meta`
**检索过程的“统计信息/说明书”**，用于展示和调试。比如：
- leaf 层级（你们系统只检索第 3 层叶子块）
- 候选数 candidate_k
- dense/sparse 各召回多少
- 是否用了 rerank（重排）
- 是否用了 auto-merge（把多个叶子块合并成更大父块）
- 是否启用了 graph（图谱上下文）

对用户最终回答不一定直接可见，但对“解释系统在干什么”非常重要。

### 4.3 `merged_context`
**已经被系统“整理拼接好”的最终参考文本**（可能=图谱知识 + 向量块文本拼接），直接拿来喂给大模型效果最好。

- 这相当于：系统替你做了“把搜索结果编排成一份阅读材料”。

---

## 5) 生成 `context`：优先用整理好的 merged_context，否则自己把 docs 拼起来
```python
context = merged if merged else _format_docs(results)
```

### 5.1 为什么优先用 merged_context？
因为 merged_context 往往已经做了更好的“排版/合并”：
- 可能包��图谱补充信息
- 可能做了去重、合并、格式化
- 大模型更容易读懂

### 5.2 如果 merged_context 为空怎么办？
就用 `_format_docs(results)` 把 `docs` 变成一段文本，格式类似：

- `[1] 文件名(Page x): 片段文本`
- `---`
- `[2] 文件名(Page y): 片段文本`

也就是把“搜索结果列表”手动拼成“可读的上下文”。

---

## 6) 再发几条 `emit_rag_step`：把关键过程讲给前端看
接下来这些 `emit_rag_step` 是“解释我做了什么”的进度提示：

### 6.1 “三级分块检索”提示
会显示你们当前在检索哪个层级（叶子层 L3）以及候选数等。citeturn0commentary0

### 6.2 “Auto-merging 合并”提示
解释是否启用了“自动合并”：
- 如果多个 L3 子块都指向同一个父块（L2/L1），系统可能用父块替换这些碎片，让上下文更完整。citeturn0commentary0

### 6.3 “检索完成”提示
告诉你找到了多少片段，以及当前检索模式（例如 hybrid / graph merge 等）。citeturn0commentary0

---

## 7) 组装 `rag_trace`：把“可解释日志”打包
```python
rag_trace = {...}
```

这一步就是把：
- 用户 query
- 找到的 chunks（results）
- 检索 meta（dense/sparse 数量、是否 rerank、graph 是否用上、auto-merge 是否发生…）

全部塞进一个字典里。用途主要是：
1) 前端展示：让用户看到“我查了哪些资料”
2) 调试排查：为什么这次没命中/命中质量差

---

## 8) return：把结果写回 state（供后续节点/生成回答使用）
最后返回的内容大意是：

- `docs`: 找到的片段列表
- `context`: 整理后的参考文本
- `rag_trace`: 过程日志

后面如果“相关性评估通过”，大模型就会用 `context` 直接生成答案；如果评估不通过，才会进入“重写查询/扩展检索”。

---

## 你可以用一句话记住 retrieve_initial 在干嘛
**把用户问题当检索词 → 从知识库捞一批相关片段 →（可能合并图谱+向量结果）→ 整理成一段“给大模型看的参考材料”→ 同时记录过程日志方便展示与调试。**


