# Local RAG MCP Server v2.1.0

基于 FastMCP + LanceDB 的 RAG 知识库 MCP 服务器，完整实现 MCP 五大核心特征 (Tools / Resources / Prompts / Sampling / Roots)。

## V2.0 核心升级

| 特性 | V1.x | V2.0 |
|------|------|------|
| 向量存储 | llama_index VectorStoreIndex | **LanceDB** (Rust 核心, mmap, 无服务器) |
| 检索算法 | AutoMergingRetriever (纯语义) | **混合检索** (IVF-PQ + BM25 + RRF 重排) |
| 文档解析 | DoclingReader (黑盒) | **自研 ETL** (PyPDF/docx/pandas, 可控可采样) |
| 分块策略 | 256/512/1024 层级分块 | **智能分块** (600char + 句尾防截断) |
| 文件监控 | 启动时 manifest 比对 | **Watchdog 实时监控** + 1.0s 防抖 |
| 日志系统 | logging + RotatingFileHandler | **Loguru** (按天轮转 + zip 压缩) |
| MCP 工具 | 3 个 (health_check/status/query) | **7 个** (完整 CRUD + 混合检索) |
| MCP 特性 | 仅 Tools | **Tools + Resources + Prompts + Sampling + Roots** |

## 项目结构

```
src/local_rag_mcp/
├── __init__.py         # 包标记
├── server.py           # FastMCP 入口 — 7 工具 + 2 资源 + 1 提示词 + Sampling + Roots
├── engine.py           # 核心引擎 — EmbeddingClient + HybridSearchEngine + IngestionPipeline
├── lancedb_store.py    # LanceDB 存储层 (IVF-PQ + BM25 FTS)
├── chunker.py          # 智能分块器 (600char/60overlap, 句尾防截断)
├── parsers/            # 多模态 ETL 解析器
│   ├── __init__.py     # 解析器工厂 get_parser()
│   ├── pdf_parser.py   # PyPDF 流式段落解析 + 高噪检测 → Sampling
│   ├── word_parser.py  # python-docx 对象树遍历 + 表格序列化
│   ├── excel_parser.py # pandas 矩阵序列化 ("列A: 值A; 列B: 值B")
│   └── text_parser.py  # MD/TXT/HTML 通用解析
├── watcher.py          # Watchdog 文件监控 + 1.0s 防抖状态机
├── sampling.py         # MCP Sampling 反向采样 (高噪文本清洗)
├── logger.py           # Loguru 按天轮转 + zip 压缩 + 10 天保留
└── status.py           # 线程安全服务状态追踪 (8 种状态)
```

## 快速开始

```powershell
# 1. 克隆项目，创建虚拟环境
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 2. 配置 .env（Embedding 地址 + 文档路径）
# 编辑 .env，关键配置:
#   EMBED_BASE_URL=http://localhost:11434
#   EMBED_MODEL_NAME=qwen3-embedding:8b
#   RAG_PROJECT_PATH=./documents

# 3. 启动
local-rag-mcp
```

## 安装与运行

```powershell
# ---- 开发模式 (源码直接跑) ----
.venv\Scripts\python.exe -m src.local_rag_mcp.server

# ---- 生产模式 (PyPI 安装) ----
pip install local-rag-mcp

# 或本地构建 wheel 后安装
.venv\Scripts\python.exe -m build
pip install dist\local_rag_mcp-2.1.0-py3-none-any.whl

# 运行 (SSE 默认, 端口 8042)
local-rag-mcp

# 运行 (stdio 模式)
$env:MCP_TRANSPORT="stdio"; local-rag-mcp

# 卸载
pip uninstall local-rag-mcp -y
```

## 接入 MCP 客户端

### SSE 模式（推荐）

先启动服务 `local-rag-mcp`，保持窗口打开：

```json
{
  "mcpServers": {
    "enterprise-local-rag": {
      "type": "sse",
      "url": "http://127.0.0.1:8042/sse"
    }
  }
}
```

### stdio 模式（子进程，客户端自动拉起）

```json
{
  "mcpServers": {
    "enterprise-local-rag": {
      "command": "local-rag-mcp",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "RAG_PROJECT_PATH": "./documents",
        "EMBED_BASE_URL": "http://localhost:11434",
        "EMBED_MODEL_NAME": "qwen3-embedding:8b"
      }
    }
  }
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_PROJECT_PATH` | (必填) | 文档项目目录，启动时自动加载索引 + 文件监控 |
| `EMBED_BASE_URL` | `http://localhost:11434` | Ollama Embedding 服务地址 |
| `EMBED_MODEL_NAME` | `bge-m3` | Embedding 模型名称 (1024 维) |
| `EMBED_TIMEOUT` | `60.0` | Embedding 请求超时（秒） |
| `MCP_TRANSPORT` | `sse` | 传输模式: `sse` (守护进程) 或 `stdio` (子进程) |
| `MCP_HOST` | `127.0.0.1` | SSE 监听地址 |
| `MCP_PORT` | `8042` | SSE 监听端口 |
| `RAG_HYBRID_WEIGHT` | `0.7` | 混合检索权重 (0=纯BM25, 1=纯向量) |
| `RAG_CHUNK_SIZE` | `600` | 分块大小（字符数） |
| `RAG_CHUNK_OVERLAP` | `60` | 块间重叠（字符数） |
| `RAG_WATCH_DEBOUNCE` | `1.0` | 文件监控防抖间隔（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## MCP 工具 (7)

| 工具 | 参数 | 说明 |
|---|---|---|
| `query_documents` | `query`, `limit?`, `scope?` | **混合检索** (密集向量 + BM25 + RRF)，返回原始文档片段 |
| `read_chunk_neighbors` | `file_id`, `chunk_id`, `direction` | 读取相邻块 (prev/next/both)，防止断章取义 |
| `ingest_file` | `path` | 索引单个文件 (PDF/Word/Excel/MD/TXT/HTML) |
| `ingest_data` | `content`, `metadata?` | 注入文本/Markdown 字符串到向量库 |
| `list_files` | 无 | 已索引文件清单及块统计 |
| `delete_file` | `file_id` | 物理擦除文件索引 + 重建 FTS |
| `status` | 无 | 系统健康诊断 (DB 统计 + Embedding + 文件监控) |

## MCP 资源 (2)

| URI | 说明 |
|---|---|
| `localrag://system/status` | 系统健康快照 (JSON): DB 统计 / Embedding / 文件监控 |
| `localrag://files/list` | 已索引文件清单 (JSON): 文件路径 + 块数量 |

## MCP 提示词 (1)

| 名称 | 参数 | 说明 |
|---|---|---|
| `review-codebase` | `focus_area` | 代码库评审 SOP 模板，自动注入 RAG 检索工作流 |

## RAG 管道 (V2.0)

1. **文件监控** — Watchdog 实时监控增删改，1.0s 防抖，自动触发 ETL
2. **多模态解析** — 自研解析器 (PyPDF/docx/pandas)，PDF 高噪自动触发 Sampling 清洗
3. **智能分块** — 600 字符块 + 60 字符重叠 + 句尾标点语义切分
4. **向量化** — Ollama `bge-m3` (1024 维)，直接调用 `/api/embeddings`
5. **双索引** — LanceDB IVF-PQ 密集向量索引 + BM25 稀疏全文倒排索引
6. **混合检索** — ANN + FTS 双路并行召回，RRF 无监督融合 (k=60, 权重可配置)

### RRF 融合公式

$$RRF\_Score(d) = \frac{w}{60 + R_{dense}(d)} + \frac{1 - w}{60 + R_{sparse}(d)}$$

- `w` = `RAG_HYBRID_WEIGHT` (默认 0.7)
- 当 w=0.7 时，兼顾深层语义并精确匹配代码/编号/专有名词

## 日志系统

### 控制台输出
彩色日志输出到 stderr，格式：`时间 | 级别 | 线程名 | 模块:行号 | 内容`

```
20:53:55 | INFO     | MainThread   | local_rag_mcp.server:67 | Enterprise RAG MCP Server v2.0 启动
20:53:55 | INFO     | MainThread   | local_rag_mcp.engine:85 | Embedding: bge-m3 @ http://localhost:11434 (dim=1024)
20:53:55 | INFO     | MainThread   | local_rag_mcp.lancedb_store:68 | LanceDB 已连接 (.ragdb_lance/)
```

### 文件归档
- **按天切分**: 每日 00:00 自动切分，文件名 `server_YYYY-MM-DD.log`
- **自动压缩**: 旧日志 zip 压缩
- **保留 10 天**: 超期日志自动物理擦除
- **全链路 Trace**: `[Trace] ▶ ...` / `[Trace] ✓ ... (XXms)` 毫秒级耗时审计

## 服务状态追踪

`status.py` 维护线程安全的全局状态单例，包含 8 种状态：

| 状态 | 说明 |
|---|---|
| `INITIALIZING` | 服务启动中 |
| `IDLE` | 空闲，等待请求 |
| `LOADING_DOCS` | 正在扫描/加载文档 |
| `INDEXING` | 正在构建向量索引 |
| `INGESTING` | 正在摄取文件/文本 |
| `QUERYING` | 正在执行混合检索 |
| `WATCHING` | 文件监控运行中 |
| `ERROR` | 发生错误（含错误信息） |

## Sampling 反向采样

当 PDF 解析器检测到高噪文本 (noise_level > 0.35) 时：
1. 后台服务挂起当前文件的写入事务
2. 通过 MCP `sampling/createMessage` 反向请求前台大模型进行文本清洗
3. 前台大模型完成推理清洗，返回结构化文本 + 核心关键词
4. 后台接收清洗结果，继续 ETL 流水线

**价值**: "白嫖"前台大模型的能力反哺本地知识库，本地服务保持轻量 (闲置物理内存 ≤ 35MB)。

## Roots 隐私隔离

当用户在编辑器中切换项目工作区时：
1. 客户端通过 `roots/list` 通知新的工作区路径
2. 服务器自动切换 LanceDB 命名空间，确保数据强隔离
3. 文件监控焦点无缝切换至新路径
4. 项目 A 的资产绝不会在项目 B 的对话中被意外召回

## 测试

```powershell
# 全部测试 (需要 Ollama 的会自动跳过)
.venv\Scripts\python.exe -m pytest tests/ -v

# 仅本地单元测试 (无需网络)
.venv\Scripts\python.exe -m pytest tests/ -v -m "not network"
```
