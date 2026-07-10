"""核心引擎测试 (需要 Ollama 服务)。

覆盖: EmbeddingClient / HybridSearchEngine / IngestionPipeline
"""

import os
import pytest

from rag_mcp.engine import (
    EmbeddingClient,
    HybridSearchEngine,
    IngestionPipeline,
    init_project,
    load_project_docs,
    query_project_docs,
    get_chunk_neighbors,
    project_context,
    is_supported_file,
)
from rag_mcp.lancedb_store import create_store

# ---- 跳过标记 (与 conftest.py 同步) ----
_ollama_url = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
_embed_model = os.getenv("EMBED_MODEL_NAME", "bge-m3")


def _check_ollama():
    try:
        import httpx
        r = httpx.get(_ollama_url, timeout=3.0)
        if r.status_code != 200:
            return False
        r = httpx.post(
            f"{_ollama_url}/api/embeddings",
            json={"model": _embed_model, "prompt": "test"},
            timeout=5.0,
        )
        return r.status_code == 200
    except Exception:
        return False


require_ollama = pytest.mark.skipif(
    not _check_ollama(),
    reason=f"Ollama Embedding 不可用 ({_ollama_url}, model={_embed_model})",
)


class TestEmbeddingClient:
    """Embedding 客户端。"""

    def test_default_config(self):
        client = EmbeddingClient()
        # 不触发 dimension 属性 (会联网) — 只验证配置值
        assert client._endpoint.endswith("/api/embeddings")
        assert client._timeout == 60.0

    def test_custom_config(self):
        client = EmbeddingClient(
            base_url="http://localhost:9999",
            model_name="custom-model",
            timeout=30.0,
        )
        assert "9999" in client._endpoint

    @require_ollama
    def test_single_embed(self):
        client = EmbeddingClient()
        vector = client.embed("测试文本 OTA 升级")
        assert len(vector) > 0
        assert len(vector) == client.dimension
        assert any(v != 0.0 for v in vector)

    @require_ollama
    def test_batch_embed(self):
        client = EmbeddingClient()
        dim = client.dimension
        texts = ["段落一", "段落二", "段落三"]
        vectors = client.embed_batch(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == dim

    def test_health_check_format(self):
        client = EmbeddingClient(base_url="http://localhost:11434")
        result = client.health_check()
        assert isinstance(result, str)
        assert "Embedding" in result


class TestHybridSearchEngine:
    """混合检索引擎 (需要已有索引)。"""

    @require_ollama
    def test_search_empty_store(self, temp_dir):
        store = create_store(str(temp_dir))
        embed = EmbeddingClient()
        engine = HybridSearchEngine(store, embed, weight=0.7)
        results = engine.search("测试查询", limit=3)
        assert isinstance(results, list)

    @require_ollama
    def test_rrf_weight_range(self, temp_dir):
        """验证权重在 0-1 范围内。"""
        store = create_store(str(temp_dir))
        embed = EmbeddingClient()
        for w in [0.0, 0.5, 1.0]:
            engine = HybridSearchEngine(store, embed, weight=w)
            results = engine.search("test", limit=2)
            assert isinstance(results, list)


class TestIngestionPipeline:
    """ETL 摄取流水线。"""

    @require_ollama
    def test_ingest_text(self, temp_dir):
        store = create_store(str(temp_dir))
        embed = EmbeddingClient()
        pipeline = IngestionPipeline(store, embed)

        result = pipeline.ingest_text(
            "OTA 升级流程需要支持断点续传和 CRC 校验。" * 10,
            metadata={"source": "test"},
            source="memory",
        )
        assert result["chunk_count"] > 0
        assert store.stats()["row_count"] > 0

    @require_ollama
    def test_ingest_txt_file(self, temp_dir, sample_txt):
        store = create_store(str(temp_dir))
        embed = EmbeddingClient()
        pipeline = IngestionPipeline(store, embed)

        result = pipeline.ingest_file(str(sample_txt))
        assert result["chunk_count"] > 0
        assert result["file_id"] == sample_txt.resolve().as_posix()

    @require_ollama
    def test_ingest_unsupported(self, temp_dir):
        store = create_store(str(temp_dir))
        embed = EmbeddingClient()
        pipeline = IngestionPipeline(store, embed)

        # 不支持的文件类型应抛出异常
        with pytest.raises(ValueError):
            pipeline.ingest_file("/tmp/test.png")


class TestProjectInit:
    """项目初始化。"""

    @require_ollama
    def test_init_project(self, temp_dir):
        result = init_project(str(temp_dir))
        assert result["project_path"] == str(temp_dir.resolve())
        assert "db_stats" in result
        assert project_context["current_path"] is not None
        assert project_context["store"] is not None
        assert project_context["search_engine"] is not None

    @require_ollama
    def test_load_project(self, sample_project):
        result = load_project_docs(str(sample_project))
        assert "索引完成" in result or "无变更" in result
        assert project_context["store"] is not None

    @require_ollama
    def test_query_after_load(self, sample_project):
        load_project_docs(str(sample_project))
        result = query_project_docs("OTA 升级")
        assert "混合检索结果" in result or "项目" in result

    def test_query_before_init(self):
        # 重置 project_context
        import rag_mcp.engine as eng
        eng.project_context["search_engine"] = None
        result = query_project_docs("test")
        assert "初始化项目" in result


class TestUtils:
    """工具函数。"""

    def test_is_supported_file(self):
        assert is_supported_file("test.pdf")
        assert is_supported_file("test.docx")
        assert is_supported_file("test.xlsx")
        assert is_supported_file("test.txt")
        assert is_supported_file("test.md")
        assert not is_supported_file("test.png")
        assert not is_supported_file("test.jpg")

    def test_get_chunk_neighbors_uninitialized(self):
        import rag_mcp.engine as eng
        eng.project_context["store"] = None
        result = get_chunk_neighbors("test.txt", 0, "next")
        assert "未初始化" in result
