"""LanceDB 嵌入式高性能向量数据库存储层。

特性:
    - 嵌入式无服务器架构 (Rust 核心, mmap 内存映射)
    - IVF-PQ 密集向量索引 + BM25 稀疏倒排索引 (FTS)
    - 无锁并发读写, 闲置物理内存 ≤ 35MB
    - 支持按文件增删改查 + 邻近块读取

Schema:
    id: str          — 唯一标识 (file_id#chunk_id)
    file_id: str     — 源文件路径
    chunk_id: int    — 块序号
    text: str        — 文本内容
    vector: list[float] — 嵌入向量 (1024 维)
    metadata: str    — JSON 序列化的元数据
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

from .logger import get_logger, TraceTimer

log = get_logger(__name__)

# 支持的文档扩展名
_SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".md", ".txt", ".html", ".htm", ".xlsx", ".xls"}


# ---------------------------------------------------------------------------
# LanceDBStore
# ---------------------------------------------------------------------------
class LanceDBStore:
    """LanceDB 嵌入式存储管理器。

    每个项目工作区拥有独立的数据库实例（通过 db_path 隔离）。
    """

    def __init__(self, db_path: str | Path, embed_dim: int = 4096):
        """初始化数据库连接。

        Args:
            db_path: LanceDB 数据目录路径 (e.g. {project_dir}/.ragdb_lance)
            embed_dim: 嵌入向量维度 (从 EmbeddingClient.dimension 获取)
        """
        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))
        self._table_name = "documents"
        self._embed_dim = embed_dim
        self._table = self._get_or_create_table()
        log.info("[LanceDB] 已连接: {} (表: {}, dim={})",
                 self._db_path, self._table_name, embed_dim)

    # ------------------------------------------------------------------
    # 表管理
    # ------------------------------------------------------------------
    def _get_or_create_table(self):
        """获取或创建文档表。"""
        table_names = self._db.table_names()
        if self._table_name in table_names:
            tbl = self._db.open_table(self._table_name)
            row_count = tbl.count_rows()
            log.info("[LanceDB] 打开已存在的表: {} ({} 行)", self._table_name, row_count)
            return tbl

        dim = self._embed_dim
        log.info("[LanceDB] 创建新表: {} ({}d vector)", self._table_name, dim)
        empty = pa.table({
            "id": pa.array([], type=pa.string()),
            "file_id": pa.array([], type=pa.string()),
            "chunk_id": pa.array([], type=pa.int32()),
            "text": pa.array([], type=pa.string()),
            "vector": pa.array([], type=pa.list_(pa.float32(), dim)),
            "metadata": pa.array([], type=pa.string()),
        })
        return self._db.create_table(self._table_name, empty, mode="overwrite")

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------
    def ensure_indexes(self):
        """确保 IVP-PQ 密集向量索引和 BM25 FTS 索引已创建。

        如果索引已存在则跳过。
        """
        # IVF-PQ 密集向量索引 (动态适配维度)
        # 每个 sub-vector 16 维: num_sub_vectors = dim / 16
        num_sub = max(1, self._embed_dim // 16)
        t0 = time.time()
        try:
            self._table.create_index(
                metric="cosine",
                num_partitions=256,
                num_sub_vectors=num_sub,
            )
            log.info("[LanceDB] IVF-PQ 索引已创建 ({:.2f}s, {} sub-vectors)",
                     time.time() - t0, num_sub)
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                log.info("[LanceDB] IVF-PQ 索引已存在，跳过")
            else:
                log.warning("[LanceDB] 索引创建: {}", e)

        # BM25 FTS 索引
        t0 = time.time()
        try:
            self._table.create_fts_index("text", replace=True)
            log.info("[LanceDB] BM25 FTS 索引已创建 ({:.2f}s)", time.time() - t0)
        except Exception as e:
            if "already exists" in str(e).lower():
                log.info("[LanceDB] FTS 索引已存在，跳过")
            else:
                log.warning("[LanceDB] FTS 索引创建: {}", e)

    # ------------------------------------------------------------------
    # CRUD 操作
    # ------------------------------------------------------------------
    def add_chunks(self, file_id: str, chunks: list[dict],
                   vectors: list[list[float]]) -> int:
        """批量写入文档块。

        Args:
            file_id: 源文件路径
            chunks: 块列表 [{"chunk_id": 0, "text": "...", "metadata": {...}}, ...]
            vectors: 对应的嵌入向量列表

        Returns:
            写入的块数量
        """
        if not chunks:
            return 0

        t0 = time.time()
        rows = []
        for i, chunk in enumerate(chunks):
            rows.append({
                "id": f"{file_id}#{chunk['chunk_id']}",
                "file_id": file_id,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "vector": vectors[i],
                "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
            })

        self._table.add(rows)
        log.info("[LanceDB] 写入 {} 个块 → {} ({:.2f}s)", len(rows), file_id, time.time() - t0)
        return len(rows)

    def delete_file(self, file_id: str) -> int:
        """从数据库物理删除指定文件的所有块。

        Args:
            file_id: 要删除的文件路径

        Returns:
            删除的行数
        """
        t0 = time.time()
        try:
            before = self._table.count_rows()
            self._table.delete(f"file_id = '{file_id}'")
            after = self._table.count_rows()
            deleted = before - after
            log.info("[LanceDB] 删除 {} 行 → {} ({:.2f}s)", deleted, file_id, time.time() - t0)
            return deleted
        except Exception:
            log.exception("[LanceDB] 删除失败: {}", file_id)
            return 0

    def list_files(self) -> list[dict]:
        """返回已索引文件清单及统计信息。

        Returns:
            [{"file_id": "...", "chunk_count": N, "last_modified": "..."}, ...]
        """
        try:
            # 使用 LanceDB 的 to_lance() 配合 PyArrow 进行聚合
            rows = self._table.search().limit(100000).to_list()
            file_stats: dict[str, dict] = {}
            for row in rows:
                fid = row["file_id"]
                if fid not in file_stats:
                    file_stats[fid] = {"file_id": fid, "chunk_count": 0}
                file_stats[fid]["chunk_count"] += 1
            return list(file_stats.values())
        except Exception:
            log.exception("[LanceDB] list_files 失败")
            return []

    def get_chunk_neighbors(self, file_id: str, chunk_id: int,
                            direction: str) -> list[dict]:
        """读取指定块的相邻块。

        Args:
            file_id: 文件路径
            chunk_id: 基准块序号
            direction: "prev" | "next" | "both"

        Returns:
            相邻块列表 [{"chunk_id": N, "text": "...", "metadata": {...}}, ...]
        """
        target_ids = []
        if direction in ("prev", "both"):
            target_ids.append(chunk_id - 1)
        if direction in ("next", "both"):
            target_ids.append(chunk_id + 1)

        target_ids = [t for t in target_ids if t >= 0]
        if not target_ids:
            return []

        try:
            # 查询相邻块
            all_rows = self._table.search().limit(100000).to_list()
            neighbors = [
                {
                    "file_id": r["file_id"],
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "metadata": json.loads(r["metadata"]) if r.get("metadata") else {},
                }
                for r in all_rows
                if r["file_id"] == file_id and r["chunk_id"] in target_ids
            ]
            neighbors.sort(key=lambda x: x["chunk_id"])
            return neighbors
        except Exception:
            log.exception("[LanceDB] get_chunk_neighbors 失败: {}#{}", file_id, chunk_id)
            return []

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search_dense(self, query_vector: list[float], limit: int = 10,
                     scope: str | None = None) -> list[dict]:
        """密集向量检索 (ANN)。

        Args:
            query_vector: 查询向量 (1024 维)
            limit: 返回结果数
            scope: 可选的 file_id 过滤关键词

        Returns:
            [{"id": ..., "file_id": ..., "chunk_id": ..., "text": ..., "_distance": ...}, ...]
        """
        try:
            q = self._table.search(query_vector).metric("cosine").limit(limit)
            if scope:
                q = q.where(f"file_id LIKE '%{scope}%'", prefilter=True)
            results = q.to_list()
            log.debug("[LanceDB] dense search: {} 结果", len(results))
            return results
        except Exception:
            log.exception("[LanceDB] dense search 失败")
            return []

    def search_sparse(self, query_text: str, limit: int = 10,
                      scope: str | None = None) -> list[dict]:
        """稀疏全文检索 (BM25 FTS)。

        Args:
            query_text: 查询关键词
            limit: 返回结果数
            scope: 可选的 file_id 过滤关键词

        Returns:
            同 search_dense 格式
        """
        try:
            # LanceDB FTS: 使用 search() with where + select
            q = self._table.search(query_text, query_type="fts").limit(limit)
            if scope:
                q = q.where(f"file_id LIKE '%{scope}%'", prefilter=True)
            results = q.to_list()
            log.debug("[LanceDB] sparse search: {} 结果", len(results))
            return results
        except Exception:
            log.exception("[LanceDB] sparse search 失败")
            return []

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """返回数据库统计信息。"""
        try:
            row_count = self._table.count_rows()
            db_size = sum(
                f.stat().st_size for f in self._db_path.rglob("*") if f.is_file()
            )
            return {
                "row_count": row_count,
                "disk_bytes": db_size,
                "disk_mb": round(db_size / (1024 * 1024), 2),
                "db_path": str(self._db_path),
            }
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def create_store(project_path: str | Path, embed_dim: int = 4096) -> LanceDBStore:
    """为指定项目创建/打开 LanceDB 存储实例。

    Args:
        project_path: 项目根目录路径
        embed_dim: 嵌入向量维度

    Returns:
        LanceDBStore 实例
    """
    db_path = Path(project_path) / ".ragdb_lance"
    store = LanceDBStore(db_path, embed_dim=embed_dim)
    store.ensure_indexes()
    return store
