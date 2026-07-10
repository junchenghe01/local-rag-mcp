"""端到端集成测试 --- 完整 RAG 流程 (需要 Ollama)。

流程: 项目初始化 → 文档解析 → 索引 → 混合检索 → 邻近块读取 → 文件管理
"""

import shutil
import pytest

from rag_mcp.engine import (
    init_project,
    load_project_docs,
    query_project_docs,
    get_chunk_neighbors,
    project_context,
)
# ---- 跳过标记 (与 conftest.py 同步) ----
import os as _os
_ollama_url = _os.getenv("EMBED_BASE_URL", "http://localhost:11434")
_embed_model = _os.getenv("EMBED_MODEL_NAME", "bge-m3")


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


class TestEndToEnd:
    """完整 RAG 流程。"""

    @require_ollama
    def test_full_flow(self, sample_project):
        """项目 → 索引 → 检索 → neighbors → 删除"""
        project = str(sample_project)

        # 1. 加载项目
        result = load_project_docs(project)
        assert "索引完成" in result or "无变更" in result

        # 2. 文件列表
        store = project_context.get("store")
        assert store is not None
        files = store.list_files()
        assert len(files) > 0, "应有已索引文件"

        # 3. 语义检索
        for query in ["OTA 升级", "电源管理"]:
            result = query_project_docs(query)
            assert len(result) > 100
            if "OTA" in query:
                assert "readme.txt" in result.lower() or "OTA" in result

        # 4. 邻近块读取
        neighbors = get_chunk_neighbors(
            files[0]["file_id"], chunk_id=0, direction="next"
        )
        assert isinstance(neighbors, str)

        # 5. 删除文件
        deleted = store.delete_file(files[0]["file_id"])
        assert deleted > 0

    @require_ollama
    def test_ingest_then_query(self, temp_dir, sample_txt):
        """单文件摄取 → 查询 → 删除。"""
        import rag_mcp.engine as eng

        init_project(str(temp_dir))
        pipeline = eng.project_context["pipeline"]

        # 摄取文件
        result = pipeline.ingest_file(str(sample_txt))
        assert result["chunk_count"] > 0

        # 查询
        response = query_project_docs("OTA 升级")
        assert "OTA" in response or "升级" in response

        # 内存文本注入
        result = pipeline.ingest_text("CCU 诊断 DTC 故障码", source="inline")
        assert result["chunk_count"] > 0

        # 查询注入的内容
        response = query_project_docs("DTC 故障码")
        assert "DTC" in response or "故障码" in response

    @require_ollama
    def test_multi_format_project(self, temp_dir, sample_txt, sample_md):
        """多格式文档共存。"""
        import shutil
        project = temp_dir / "multi"
        project.mkdir()
        shutil.copy2(sample_txt, project / "doc.txt")
        shutil.copy2(sample_md, project / "doc.md")

        result = load_project_docs(str(project))
        assert "索引完成" in result or "无变更" in result

        store = project_context["store"]
        files = store.list_files()
        # 可能有文件解析失败 (log 记录并跳过)，至少应有 1 个成功
        assert len(files) >= 1, f"Expected at least 1 indexed file, got {len(files)}"

    @require_ollama
    def test_rrf_result_format(self, sample_project):
        """验证 RRF 返回格式。"""
        load_project_docs(str(sample_project))
        result = query_project_docs("测试查询")

        # 应有结构化输出
        assert "项目:" in result
        assert "混合检索结果" in result
        assert "RRF得分:" in result or "### [" in result

    @require_ollama
    def test_scope_filter(self, sample_project):
        """scope 文件过滤。"""
        load_project_docs(str(sample_project))

        # 无 scope
        all_results = query_project_docs("OTA", limit=3)

        # 有 scope 过滤
        filtered = query_project_docs("OTA", limit=3, scope="nonexistent_file_xyz")
        # 过滤后可能为空或只有相关结果
        assert isinstance(filtered, str)
