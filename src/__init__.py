"""RAG MCP Server V2.0 — 基于 FastMCP + LanceDB 的企业级文档检索引擎。

完整实现 MCP 五大核心特征:
    - Tools (7): ingest_file, ingest_data, query_documents, read_chunk_neighbors,
                 list_files, delete_file, status
    - Resources (2): localrag://system/status, localrag://files/list
    - Prompts (1): review-codebase
    - Sampling: PDF 高噪文本反向清洗
    - Roots: 动态工作区隔离
"""
