# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Knowledge MCP Server — FastMCP + LanceDB + Hybrid Search, full MCP 5 features (Tools/Resources/Prompts/Sampling/Roots).

Code protection via Cython (.pyd) for release builds.

## Commands

```bash
.\.venv\Scripts\python.exe -m src.rag_mcp.server              # 开发模式启动
.\.venv\Scripts\python.exe build\build_release.py             # 构建 wheel (Cython .pyd)
.\.venv\Scripts\python.exe build\build_release.py --install   # 构建 + 安装
.\.venv\Scripts\python.exe build\build_release.py --no-cython # 跳过 Cython
.\.venv\Scripts\python.exe -m pip install -e .                 # 可编辑安装
.\.venv\Scripts\python.exe -m pytest tests/ -v                 # 运行测试
```

## Architecture

```
src/rag_mcp/
├── __init__.py       # Package marker
├── server.py         # FastMCP entry — 7 tools + 2 resources + 1 prompt + Sampling + Roots
├── engine.py         # [Cython target] Core: EmbeddingClient, HybridSearchEngine, IngestionPipeline
├── lancedb_store.py  # LanceDB storage layer (IVF-PQ + BM25 FTS)
├── chunker.py        # Smart chunking (600char/60overlap, sentence boundary)
├── parsers/          # Multi-modal ETL parsers
│   ├── __init__.py
│   ├── pdf_parser.py     # PyPDF paragraph reconstruction
│   ├── word_parser.py    # python-docx object tree traversal
│   ├── excel_parser.py   # pandas matrix serialization (20 rows/section)
│   └── text_parser.py    # MD/TXT/HTML text extraction
├── watcher.py        # Watchdog file monitoring + 1.0s debounce
├── sampling.py       # Sampling callback for noisy document cleaning
├── logger.py         # Loguru-based structured logging (daily rotation)
└── status.py         # Thread-safe server status tracking

build/
└── build_release.py   # Wheel build (Cython .pyd + wheel packaging)

tests/                  # Test suite (80 tests)
```

## RAG Pipeline (v2.1)

1. **Document Parsing**: Custom parsers (PyPDF/python-docx/pandas) extract text
2. **Smart Chunking**: 600 char chunks, 60 char overlap, sentence boundary detection
3. **Embedding**: Direct Ollama `/api/embeddings` call, auto-detect dimension
4. **Storage**: LanceDB embedded vector DB with IVF-PQ dense index + BM25 FTS
5. **Search**: Hybrid ANN + BM25 dual recall, RRF (Reciprocal Rank Fusion) merge
6. **File Watch**: Watchdog monitors file changes, auto re-index with 1.0s debounce

## MCP Features

**7 Tools**: `ingest_file`, `ingest_data`, `query_documents`, `read_chunk_neighbors`, `list_files`, `delete_file`, `status`

**2 Resources**: `localrag://system/status`, `localrag://files/list`

**1 Prompt**: `review-codebase`

**Sampling**: Reverse sampling for noisy PDF text cleaning

**Roots**: Dynamic workspace isolation

## Configuration

All settings in `.env` (not committed), loaded via `python-dotenv`:

| Variable | Default |
|---|---|
| `EMBED_BASE_URL` | `http://localhost:11434` |
| `EMBED_MODEL_NAME` | `bge-m3` |
| `RAG_HYBRID_WEIGHT` | `0.7` |
| `RAG_CHUNK_SIZE` | `600` |
| `RAG_CHUNK_OVERLAP` | `60` |
| `RAG_WATCH_DEBOUNCE` | `1.0` |
| `MCP_PORT` | `8042` |
