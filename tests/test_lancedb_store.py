"""LanceDB 存储层测试。

覆盖: 表创建 / 块写入 / 检索 / 删除 / 邻近块 / 文件列表
"""

import pytest
from local_rag_mcp.lancedb_store import LanceDBStore, create_store

# 测试用向量维度 (与 qwen3-embedding:8b 一致)
TEST_DIM = 4096


class TestLanceDBStore:
    """CRUD 操作测试。"""

    @pytest.fixture
    def store(self, temp_dir):
        db_dir = temp_dir / ".ragdb_test"
        s = LanceDBStore(db_dir, embed_dim=TEST_DIM)
        s.ensure_indexes()
        yield s

    def test_create_store(self, temp_dir):
        s = create_store(str(temp_dir), embed_dim=TEST_DIM)
        assert s.stats()["row_count"] >= 0

    def test_add_and_count(self, store):
        chunks = [
            {"chunk_id": 0, "text": "这是第一段测试文本", "metadata": {}},
            {"chunk_id": 1, "text": "这是第二段测试文本", "metadata": {}},
        ]
        vectors = [[0.1] * TEST_DIM, [0.2] * TEST_DIM]
        store.add_chunks("test/file.txt", chunks, vectors)
        stats = store.stats()
        assert stats["row_count"] == 2

    def test_add_empty(self, store):
        assert store.add_chunks("test/empty.txt", [], []) == 0

    def test_delete_file(self, store):
        chunks = [{"chunk_id": 0, "text": "临时内容", "metadata": {}}]
        vectors = [[0.3] * TEST_DIM]
        store.add_chunks("test/temp.txt", chunks, vectors)
        assert store.stats()["row_count"] == 1
        deleted = store.delete_file("test/temp.txt")
        assert deleted == 1
        assert store.stats()["row_count"] == 0

    def test_delete_nonexistent(self, store):
        assert store.delete_file("nonexistent/file.txt") == 0

    def test_list_files(self, store):
        store.add_chunks("a/doc1.txt",
                         [{"chunk_id": 0, "text": "a", "metadata": {}}],
                         [[0.1] * TEST_DIM])
        store.add_chunks("b/doc2.txt",
                         [{"chunk_id": 0, "text": "b", "metadata": {}},
                          {"chunk_id": 1, "text": "c", "metadata": {}}],
                         [[0.2] * TEST_DIM, [0.3] * TEST_DIM])
        files = store.list_files()
        assert len(files) == 2
        counts = {f["file_id"]: f["chunk_count"] for f in files}
        assert counts.get("a/doc1.txt") == 1
        assert counts.get("b/doc2.txt") == 2

    def test_get_chunk_neighbors(self, store):
        chunks = [
            {"chunk_id": i, "text": f"段落{i}", "metadata": {}}
            for i in range(5)
        ]
        vectors = [[0.1] * TEST_DIM] * 5
        store.add_chunks("test/neighbors.txt", chunks, vectors)

        # prev
        prev = store.get_chunk_neighbors("test/neighbors.txt", 3, "prev")
        assert len(prev) == 1
        assert prev[0]["chunk_id"] == 2

        # next
        nxt = store.get_chunk_neighbors("test/neighbors.txt", 3, "next")
        assert len(nxt) == 1
        assert nxt[0]["chunk_id"] == 4

        # both
        both = store.get_chunk_neighbors("test/neighbors.txt", 3, "both")
        assert len(both) == 2
        ids = [b["chunk_id"] for b in both]
        assert 2 in ids and 4 in ids

        # boundary: chunk 0 没有 prev
        first = store.get_chunk_neighbors("test/neighbors.txt", 0, "prev")
        assert len(first) == 0

    def test_dense_search(self, store):
        chunks = [
            {"chunk_id": 0, "text": "OTA 远程升级系统", "metadata": {}},
            {"chunk_id": 1, "text": "电源管理模块", "metadata": {}},
            {"chunk_id": 2, "text": "诊断功能 DTC", "metadata": {}},
        ]
        D = TEST_DIM
        vectors = [
            [0.9] + [0.0] * (D - 1),
            [0.0, 0.9] + [0.0] * (D - 2),
            [0.0, 0.0, 0.9] + [0.0] * (D - 3),
        ]
        store.add_chunks("test/search.txt", chunks, vectors)

        results = store.search_dense([0.95] + [0.0] * (D - 1), limit=2)
        assert len(results) <= 2
        if results:
            assert "file_id" in results[0]
            assert "text" in results[0]
            assert "_distance" in results[0]

    def test_sparse_search(self, store):
        chunks = [
            {"chunk_id": 0, "text": "OTA 远程升级功能需求", "metadata": {}},
            {"chunk_id": 1, "text": "电源管理 低功耗 睡眠模式", "metadata": {}},
        ]
        vectors = [[0.1] * TEST_DIM, [0.2] * TEST_DIM]
        store.add_chunks("test/fts.txt", chunks, vectors)

        results = store.search_sparse("OTA 升级", limit=2)
        assert isinstance(results, list)

    def test_metadata_json(self, store):
        meta = {"author": "test", "tags": ["ota", "ccu"]}
        chunks = [{"chunk_id": 0, "text": "带元数据的文本", "metadata": meta}]
        store.add_chunks("test/meta.json", chunks, [[0.1] * TEST_DIM])

        import json
        results = store.search_dense([0.1] * TEST_DIM, limit=1)
        if results:
            stored_meta = json.loads(results[0].get("metadata", "{}"))
            assert stored_meta == meta

    def test_stats(self, store):
        s = store.stats()
        assert "row_count" in s
        assert "disk_mb" in s
        assert "db_path" in s
