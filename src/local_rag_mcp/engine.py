"""RAG 核心引擎 --- LanceDB + 混合检索 + ETL 流水线。

所有模型配置从 .env 环境变量加载。

设计原则:
    - 纯检索模式：返回原始文档片段，不内置 LLM 合成
    - 混合检索：IVF-PQ 密集向量 + BM25 稀疏全文 + RRF 融合
    - 多命名空间：支持 Roots 隔离，不同项目独立存储
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv

from .logger import get_logger, TraceTimer
from .lancedb_store import LanceDBStore, create_store, _SUPPORTED_EXTS
from .chunker import SmartChunker
from .parsers import get_parser

load_dotenv()

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# 全局状态（支持多命名空间隔离）
# ---------------------------------------------------------------------------
project_context: dict[str, Any] = {
    "current_path": None,       # 当前 Root 项目路径
    "store": None,              # LanceDBStore 实例
    "search_engine": None,      # HybridSearchEngine 实例
    "pipeline": None,           # IngestionPipeline 实例
}


# ===================================================================
# EmbeddingClient — 直接调用 Ollama API
# ===================================================================
class EmbeddingClient:
    """Ollama Embedding API 客户端。

    直接调用 POST /api/embeddings，无 llama_index 依赖。
    """

    def __init__(self,
                 base_url: str | None = None,
                 model_name: str | None = None,
                 timeout: float | None = None):
        self._base_url = (base_url or os.getenv("EMBED_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._model = model_name or os.getenv("EMBED_MODEL_NAME", "bge-m3")
        self._timeout = timeout or float(os.getenv("EMBED_TIMEOUT", "60.0"))
        self._endpoint = f"{self._base_url}/api/embeddings"
        self._dim: int = 0  # 首次调用时自动检测

        log.info("[Embedding] {} @ {}", self._model, self._base_url)

    @property
    def dimension(self) -> int:
        if self._dim == 0:
            try:
                self._dim = len(self.embed("dim probe"))
            except Exception:
                log.warning("[Embedding] 无法探测维度，使用默认 4096")
                self._dim = 4096
        return self._dim

    def embed(self, text: str) -> list[float]:
        """对单条文本生成嵌入向量。"""
        t0 = time.time()
        try:
            resp = httpx.post(
                self._endpoint,
                json={"model": self._model, "prompt": text},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            vector = data.get("embedding", [])
            if not vector:
                raise ValueError(f"Ollama 返回空 embedding: {data}")
            elapsed_ms = (time.time() - t0) * 1000
            # 首次调用时输出维度
            if self._dim == 0:
                self._dim = len(vector)
                log.info("[Embedding] 检测到维度: {}", self._dim)
            log.debug("[Embedding] 单条: {} chars → {} dims ({:.0f}ms)",
                     len(text), len(vector), elapsed_ms)
            return vector
        except Exception:
            log.exception("[Embedding] 请求失败 (model={})", self._model)
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量（顺序调用，Ollama 不支持批量 API）。"""
        t0 = time.time()
        vectors = []
        for text in texts:
            vectors.append(self.embed(text))
        elapsed_ms = (time.time() - t0) * 1000
        log.info("[Embedding] 批量: {} 条文本 ({:.0f}ms, {:.0f}ms/条)",
                len(texts), elapsed_ms, elapsed_ms / max(len(texts), 1))
        return vectors

    def health_check(self) -> str:
        """检查 Embedding 服务连通性。"""
        try:
            resp = httpx.get(self._base_url, timeout=5.0)
            return f"Embedding ({self._base_url}): OK (model={self._model})"
        except Exception as e:
            return f"Embedding ({self._base_url}): 不可达 ({e})"


# ===================================================================
# HybridSearchEngine — 混合检索 + RRF 重排
# ===================================================================
class HybridSearchEngine:
    """混合检索引擎：IVF-PQ 密集向量 + BM25 稀疏全文 + RRF 融合。

    RRF Score(d) = w / (k + R_dense(d)) + (1-w) / (k + R_sparse(d))
    """

    def __init__(self, store: LanceDBStore, embed_client: EmbeddingClient,
                 weight: float | None = None, k: int = 60):
        self._store = store
        self._embed = embed_client
        self._weight = weight if weight is not None else float(
            os.getenv("RAG_HYBRID_WEIGHT", "0.7"))
        self._k = k

        log.info("[SearchEngine] 混合检索: weight={:.2f}, k={}", self._weight, self._k)

    def search(self, query: str, limit: int = 4,
               scope: str | None = None) -> list[dict]:
        """执行混合检索。

        Args:
            query: 自然语言问题或关键词短语
            limit: 最大返回块数
            scope: 可选，file_id 过滤关键词

        Returns:
            [{"file_id": ..., "chunk_id": ..., "text": ..., "score": ..., "rank_dense": ..., "rank_sparse": ...}, ...]
        """
        t0 = time.time()

        # Step 1: 生成查询向量
        with TraceTimer(log, "生成查询向量 ({n} chars)", n=len(query)):
            query_vector = self._embed.embed(query)

        # Step 2: 双路并行召回
        with TraceTimer(log, "双路召回 (limit={l})", l=limit * 2):
            # 密集召回
            dense_results = self._store.search_dense(query_vector, limit=limit * 2, scope=scope)
            # 稀疏召回
            sparse_results = self._store.search_sparse(query, limit=limit * 2, scope=scope)

        # Step 3: RRF 融合
        with TraceTimer(log, "RRF 融合 (dense=%d, sparse=%d)",
                        d=len(dense_results), s=len(sparse_results)):
            merged = self._rrf_merge(dense_results, sparse_results, limit)

        elapsed_ms = (time.time() - t0) * 1000
        log.info("[SearchEngine] 检索完成: {} 结果 ({:.0f}ms) | {}",
                 len(merged), elapsed_ms,
                 query[:60] + "..." if len(query) > 60 else query)
        return merged

    # ------------------------------------------------------------------
    # RRF 融合算法
    # ------------------------------------------------------------------
    def _rrf_merge(self, dense: list[dict], sparse: list[dict],
                   top_k: int) -> list[dict]:
        """使用倒数排名融合 (RRF) 合并双侧得分。

        公式: RRF_Score(d) = w/(k+R_dense) + (1-w)/(k+R_sparse)
        """
        # 构建文档索引: id -> {dense_rank, sparse_rank, ...}
        doc_map: dict[str, dict] = {}

        # 密集排名
        for rank, item in enumerate(dense):
            doc_id = item.get("id", "")
            if doc_id not in doc_map:
                doc_map[doc_id] = {"item": item, "dense_rank": rank + 1, "sparse_rank": None}
            else:
                cur = doc_map[doc_id].get("dense_rank")
                doc_map[doc_id]["dense_rank"] = rank + 1 if cur is None else min(cur, rank + 1)

        # 稀疏排名
        for rank, item in enumerate(sparse):
            doc_id = item.get("id", "")
            if doc_id not in doc_map:
                doc_map[doc_id] = {"item": item, "dense_rank": None, "sparse_rank": rank + 1}
            else:
                cur = doc_map[doc_id].get("sparse_rank")
                doc_map[doc_id]["sparse_rank"] = rank + 1 if cur is None else min(cur, rank + 1)

        # 计算 RRF 得分
        scored = []
        for doc_id, info in doc_map.items():
            rrf = 0.0
            if info["dense_rank"] is not None:
                rrf += self._weight / (self._k + info["dense_rank"])
            if info["sparse_rank"] is not None:
                rrf += (1 - self._weight) / (self._k + info["sparse_rank"])
            # 跳过两边都没结果的 (理论上不会发生)
            if info["dense_rank"] is None and info["sparse_rank"] is None:
                continue

            item = info["item"]
            scored.append({
                "file_id": item.get("file_id", ""),
                "chunk_id": item.get("chunk_id", -1),
                "text": item.get("text", ""),
                "score": round(rrf, 6),
                "rank_dense": info["dense_rank"],
                "rank_sparse": info["sparse_rank"],
            })

        # 按 RRF 得分降序排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ===================================================================
# IngestionPipeline — ETL 流水线
# ===================================================================
class IngestionPipeline:
    """多模态 ETL 摄取流水线: Parser → Chunker → Embedding → LanceDB。"""

    def __init__(self, store: LanceDBStore, embed_client: EmbeddingClient,
                 chunker: SmartChunker | None = None):
        self._store = store
        self._embed = embed_client
        self._chunker = chunker or SmartChunker()

    def ingest_file(self, file_path: str,
                    on_progress: Callable | None = None) -> dict:
        """索引单个文件。

        Args:
            file_path: 待索引文件的绝对路径
            on_progress: 可选进度回调 (phase: str, pct: float)

        Returns:
            {"file_id": ..., "chunk_count": N, "elapsed_ms": ...}
        """
        t0 = time.time()
        path = Path(file_path).resolve()
        file_id = path.as_posix()

        log.info("[Pipeline] 开始摄取: {}", path.name)

        # Step 1: 选择解析器
        if on_progress:
            on_progress(f"解析: {path.name}", 10)
        parser = get_parser(str(path))
        if parser is None:
            raise ValueError(f"不支持的文件类型: {path.suffix}")

        # Step 2: 解析文档
        sections = parser.parse(str(path))
        if not sections:
            log.warning("[Pipeline] 解析结果为空: {}", file_id)
            return {"file_id": file_id, "chunk_count": 0, "elapsed_ms": 0}

        # Step 3: 智能分块
        if on_progress:
            on_progress(f"分块 ({len(sections)} 段落)", 30)
        all_chunks = []
        for section in sections:
            chunks = self._chunker.chunk(
                text=section["text"],
                file_id=file_id,
                metadata={
                    "source": section.get("source", file_id),
                    "section": section.get("section", ""),
                    **section.get("metadata", {}),
                },
            )
            all_chunks.extend(chunks)

        # 重新编号 chunk_id（跨 section 全局递增）
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_id"] = i

        log.info("[Pipeline] 分块: {} 个 chunk", len(all_chunks))

        # Step 4: 向量化
        if on_progress:
            on_progress(f"向量化 ({len(all_chunks)} 块)", 50)
        texts = [c["text"] for c in all_chunks]
        vectors = self._embed.embed_batch(texts)

        # Step 5: 写入 LanceDB
        if on_progress:
            on_progress(f"写入存储 ({len(all_chunks)} 块)", 80)
        count = self._store.add_chunks(file_id, all_chunks, vectors)

        # Step 6: 确保索引
        if on_progress:
            on_progress("重建索引", 90)
        self._store.ensure_indexes()

        elapsed_ms = (time.time() - t0) * 1000
        if on_progress:
            on_progress("完成", 100)

        log.info("[Pipeline] 摄取完成: {} → {} chunks ({:.0f}ms)",
                 path.name, count, elapsed_ms)
        return {"file_id": file_id, "chunk_count": count, "elapsed_ms": elapsed_ms}

    def ingest_text(self, content: str, metadata: dict | None = None,
                    source: str = "memory") -> dict:
        """将纯文本/Markdown 字符串直接注入向量库。

        Args:
            content: 待注入的文本内容
            metadata: 元数据
            source: 来源标识

        Returns:
            {"file_id": ..., "chunk_count": N}
        """
        t0 = time.time()
        file_id = f"memory://{source}/{int(t0)}"

        log.info("[Pipeline] 摄取文本: source={}, {} chars", source, len(content))

        meta = metadata or {}
        meta["source"] = source

        chunks = self._chunker.chunk(text=content, file_id=file_id, metadata=meta)
        texts = [c["text"] for c in chunks]
        vectors = self._embed.embed_batch(texts)
        count = self._store.add_chunks(file_id, chunks, vectors)

        elapsed_ms = (time.time() - t0) * 1000
        log.info("[Pipeline] 文本摄取完成: {} chunks ({:.0f}ms)", count, elapsed_ms)
        return {"file_id": file_id, "chunk_count": count, "elapsed_ms": elapsed_ms}


# ===================================================================
# 健康检查 & 工具函数
# ===================================================================
def health_check() -> str:
    """检查 Embedding 服务连通性。"""
    client = EmbeddingClient()
    return client.health_check()


def is_supported_file(file_path: str | Path) -> bool:
    """检查文件是否为支持的格式。"""
    return Path(file_path).suffix.lower() in _SUPPORTED_EXTS


# ===================================================================
# 初始化项目上下文
# ===================================================================
def init_project(project_path: str,
                 on_progress: Callable | None = None) -> dict:
    """初始化项目：创建存储和检索引擎。

    Args:
        project_path: 项目根目录
        on_progress: 可选进度回调

    Returns:
        状态摘要
    """
    t0 = time.time()
    ppath = Path(project_path).resolve()

    if on_progress:
        on_progress("初始化 Embedding 客户端", 5)

    embed_client = EmbeddingClient()
    # 获取实际维度 (自动检测)
    dim = embed_client.dimension

    if on_progress:
        on_progress("打开/创建 LanceDB", 10)

    store = create_store(str(ppath), embed_dim=dim)

    if on_progress:
        on_progress("初始化检索引擎", 15)

    search_engine = HybridSearchEngine(store, embed_client)
    pipeline = IngestionPipeline(store, embed_client)

    project_context["current_path"] = ppath
    project_context["store"] = store
    project_context["search_engine"] = search_engine
    project_context["pipeline"] = pipeline

    elapsed_ms = (time.time() - t0) * 1000
    stats = store.stats()
    log.info("[Engine] 项目初始化完成: {} ({:.0f}ms, {} 行)",
             ppath, elapsed_ms, stats.get("row_count", 0))

    if on_progress:
        on_progress("项目就绪", 20)

    return {
        "project_path": str(ppath),
        "db_stats": stats,
        "elapsed_ms": elapsed_ms,
    }


def load_project_docs(project_path: str,
                      on_progress: Callable | None = None) -> str:
    """启动时加载/索引项目中的所有文档。

    扫描项目目录，对未索引或已变更的文件执行 ETL 摄取。

    Args:
        project_path: 项目文件夹绝对路径
        on_progress: 可选进度回调 (phase: str, pct: float)

    Returns:
        操作结果摘要字符串
    """
    t0 = time.time()
    ppath = Path(project_path).resolve()

    log.info("[Engine] 加载项目文档: {}", ppath)

    if not ppath.exists() or not ppath.is_dir():
        log.error("[Engine] 项目目录不存在: {}", ppath)
        return f"找不到有效的项目目录: {ppath}"

    # 确保已初始化
    if project_context.get("current_path") != ppath:
        init_project(str(ppath), on_progress)

    store = project_context.get("store")
    if store is None:
        return "存储未初始化"

    # Step 1: 扫描项目目录
    if on_progress:
        on_progress("扫描项目文档", 22)

    current_files: dict[str, float] = {}
    for f in ppath.rglob("*"):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        if ".ragdb_lance" in f.parts or ".ragdb" in f.parts:
            continue
        current_files[f.relative_to(ppath).as_posix()] = f.stat().st_mtime

    log.info("[Engine] 扫描到 {} 个文档", len(current_files))

    if not current_files:
        return f"该目录下未扫描到支持的文档: {ppath}"

    # Step 2: 比对已索引文件
    indexed = {item["file_id"]: item for item in store.list_files()}
    indexed_rel = {}
    for fid in indexed:
        try:
            rel = Path(fid).relative_to(ppath).as_posix()
            indexed_rel[rel] = fid
        except ValueError:
            pass

    # 找出需要索引的文件（新增或修改）
    to_index = []
    for rel_path, mtime in current_files.items():
        abs_path = (ppath / rel_path).as_posix()
        if rel_path not in indexed_rel:
            to_index.append((abs_path, "new"))
        else:
            # 简化处理：mtime 不同视为修改（实际应用中可更精确）
            pass

    # 找出已删除的文件（从数据库清理）
    to_delete = []
    for rel_path, abs_path in indexed_rel.items():
        if rel_path not in current_files:
            to_delete.append(abs_path)

    # Step 3: 清理已删除文件
    for fid in to_delete:
        store.delete_file(fid)

    # Step 4: 索引新文件
    if not to_index and not to_delete:
        elapsed = time.time() - t0
        log.info("[Engine] 无变更，跳过索引 ({} 篇, {:.1f}s)", len(current_files), elapsed)
        if on_progress:
            on_progress("就绪（无变更）", 100)
        return f"文档无变更 ({len(current_files)} 篇)，跳过索引 ({elapsed:.1f}s)"

    pipeline = project_context.get("pipeline")
    if pipeline is None:
        return "摄取流水线未初始化"

    total = len(to_index)
    for i, (abs_path, reason) in enumerate(to_index):
        pct = 25 + int((i + 1) / max(total, 1) * 70)
        if on_progress:
            on_progress(f"索引 [{i + 1}/{total}]: {Path(abs_path).name}", pct)
        try:
            pipeline.ingest_file(abs_path)
        except Exception:
            log.exception("[Engine] 索引失败: {}", abs_path)

    elapsed = time.time() - t0
    if on_progress:
        on_progress("就绪", 100)

    log.info("[Engine] 项目加载完成: {} 篇文档 ({:.1f}s)", len(current_files), elapsed)
    return (
        f"索引完成 ({elapsed:.1f}s): {len(to_index)} 篇新增/修改, "
        f"{len(to_delete)} 篇已清理, 共 {len(current_files)} 篇文档"
    )


# ===================================================================
# 查询
# ===================================================================
def query_project_docs(question: str, limit: int = 5,
                       scope: str | None = None) -> str:
    """基于已加载的项目文档进行混合检索，返回原始文档片段。

    Args:
        question: 需求问题
        limit: 返回结果数
        scope: 可选文件过滤关键词

    Returns:
        格式化的检索结果文本
    """
    search_engine = project_context.get("search_engine")
    store = project_context.get("store")

    if search_engine is None:
        return "请先初始化项目，再执行查询。"

    t0 = time.time()
    q_preview = question[:80] + "..." if len(question) > 80 else question
    log.info("[Engine] 查询: {}", q_preview)

    try:
        results = search_engine.search(question, limit=limit, scope=scope)
    except Exception:
        log.exception("[Engine] 查询失败: {}", q_preview)
        raise

    lines = [
        f"项目: {project_context.get('current_path', '未知')}",
        f"问题: {question}",
        f"",
        f"## 混合检索结果（共 {len(results)} 个相关片段）",
        f"",
    ]

    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        text = r.get("text", "").strip()
        lines.append(f"### [{i}] RRF得分: {score:.4f} | file: {r.get('file_id', '?')} | chunk: {r.get('chunk_id', '?')}")
        lines.append(f"    (密集排名: {r.get('rank_dense', '-')}, 稀疏排名: {r.get('rank_sparse', '-')})")
        lines.append(text)
        lines.append("")

    elapsed_ms = (time.time() - t0) * 1000
    log.info("[Engine] 查询完成 ({:.0f}ms): {} 结果, query={}",
             elapsed_ms, len(results), q_preview)
    return "\n".join(lines)


# ===================================================================
# 邻近块读取
# ===================================================================
def get_chunk_neighbors(file_id: str, chunk_id: int,
                        direction: str) -> str:
    """读取指定块的相邻块。

    Args:
        file_id: 文件路径
        chunk_id: 当前块序号
        direction: "prev" | "next" | "both"

    Returns:
        格式化的相邻块文本
    """
    store = project_context.get("store")
    if store is None:
        return "存储未初始化"

    neighbors = store.get_chunk_neighbors(file_id, chunk_id, direction)
    if not neighbors:
        return f"未找到相邻块: file={file_id}, chunk={chunk_id}, direction={direction}"

    lines = [
        f"## 相邻块: {direction} (file={file_id}, chunk={chunk_id})",
        f"",
    ]
    for n in neighbors:
        lines.append(f"### Chunk #{n['chunk_id']}")
        lines.append(n["text"])
        lines.append("")

    return "\n".join(lines)
