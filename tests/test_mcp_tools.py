"""MCP 工具注册与 Schema 测试。

覆盖: 7 个工具的注册 / 参数 Schema / Resources / Prompts
"""

import asyncio
import pytest

from local_rag_mcp.server import mcp


def _run(coro):
    """同步运行 async 函数。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


class TestToolRegistration:
    """工具注册测试。"""

    def test_all_7_tools_registered(self):
        tools = _run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        expected = {
            "ingest_file", "ingest_data", "query_documents",
            "read_chunk_neighbors", "list_files", "delete_file", "status",
        }
        assert tool_names == expected, f"Missing: {expected - tool_names}, Extra: {tool_names - expected}"

    def test_query_documents_schema(self):
        tools = _run(mcp.list_tools())
        qt = next(t for t in tools if t.name == "query_documents")
        schema = qt.inputSchema
        assert "query" in schema.get("properties", {})
        assert "limit" in schema.get("properties", {})
        assert "scope" in schema.get("properties", {})
        assert "query" in schema.get("required", [])

    def test_ingest_file_schema(self):
        tools = _run(mcp.list_tools())
        it = next(t for t in tools if t.name == "ingest_file")
        schema = it.inputSchema
        assert "path" in schema.get("properties", {})
        assert "path" in schema.get("required", [])

    def test_ingest_data_schema(self):
        tools = _run(mcp.list_tools())
        it = next(t for t in tools if t.name == "ingest_data")
        schema = it.inputSchema
        assert "content" in schema.get("properties", {})
        assert "content" in schema.get("required", [])

    def test_read_chunk_neighbors_schema(self):
        tools = _run(mcp.list_tools())
        rt = next(t for t in tools if t.name == "read_chunk_neighbors")
        schema = rt.inputSchema
        props = schema.get("properties", {})
        assert "file_id" in props
        assert "chunk_id" in props
        assert "direction" in props
        # direction 应限制为 prev/next/both
        direction = props.get("direction", {})
        if "enum" in direction:
            assert set(direction["enum"]) == {"prev", "next", "both"}

    def test_delete_file_schema(self):
        tools = _run(mcp.list_tools())
        dt = next(t for t in tools if t.name == "delete_file")
        schema = dt.inputSchema
        assert "file_id" in schema.get("properties", {})

    def test_no_param_tools(self):
        """status / list_files 无必填参数。"""
        tools = _run(mcp.list_tools())
        for name in ("status", "list_files"):
            t = next(t for t in tools if t.name == name)
            req = t.inputSchema.get("required", [])
            assert req == [] or req is None


class TestResources:
    """Resources 注册测试。"""

    def test_resources_registered(self):
        resources = _run(mcp.list_resources())
        uris = {str(r.uri) for r in resources}
        assert "localrag://system/status" in uris
        assert "localrag://files/list" in uris


class TestPrompts:
    """Prompts 注册测试。"""

    def test_prompt_registered(self):
        prompts = _run(mcp.list_prompts())
        names = {p.name for p in prompts}
        assert "review_codebase" in names

    def test_prompt_has_param(self):
        prompts = _run(mcp.list_prompts())
        rp = next(p for p in prompts if p.name == "review_codebase")
        args = {a.name for a in (rp.arguments or [])}
        assert "focus_area" in args


class TestServerInfo:
    """服务信息测试。"""

    def test_server_name(self):
        assert "Enterprise" in mcp.name or "RAG" in mcp.name

    def test_status_module(self):
        from local_rag_mcp.status import ServerState, server_status
        assert ServerState.IDLE is not None
        snap = server_status.to_dict()
        assert "status" in snap
