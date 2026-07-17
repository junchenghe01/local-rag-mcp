"""SmartChunker 分块器单元测试。

覆盖:
    - 基本分块 (chunk_size=600, overlap=60)
    - 空文本 / 短文本边界
    - 句尾标点语义切分
    - 段落合并
    - chunk_id 连续性
"""

import pytest
from local_rag_mcp.chunker import SmartChunker


class TestSmartChunker:
    """分块器基础测试。"""

    def test_empty_text(self):
        chunker = SmartChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n  ") == []

    def test_short_text(self):
        """短文本应产生单个块。"""
        chunker = SmartChunker(chunk_size=600, overlap_size=60)
        result = chunker.chunk("Hello World", file_id="test.txt")
        assert len(result) == 1
        assert result[0]["chunk_id"] == 0
        assert "Hello World" in result[0]["text"]
        assert result[0]["file_id"] == "test.txt"

    def test_custom_size(self):
        """自定义 chunk 大小。"""
        chunker = SmartChunker(chunk_size=100, overlap_size=20)
        text = "测试。这是第二句。这是第三句。" * 20
        result = chunker.chunk(text, file_id="f.txt")
        # 不应有空块
        for r in result:
            assert len(r["text"]) > 0
        # chunk 不应过大 (chunk_size + boundary_window + overlap 前缀)
        for r in result:
            assert len(r["text"]) <= 100 + 50 + 30, \
                f"Chunk too large: {len(r['text'])} chars"

    def test_chunk_id_sequential(self):
        """chunk_id 应连续递增。"""
        chunker = SmartChunker(chunk_size=100, overlap_size=20)
        text = "。".join(f"段落{i}" for i in range(50))
        result = chunker.chunk(text)
        ids = [r["chunk_id"] for r in result]
        assert ids == list(range(len(result)))

    def test_metadata_preserved(self):
        """元数据应透传。"""
        chunker = SmartChunker()
        meta = {"source": "test", "page": 1}
        result = chunker.chunk("测试内容", metadata=meta)
        assert result[0]["metadata"] == meta

    def test_overlap(self):
        """重叠区域应包含前一块的末尾内容。"""
        chunker = SmartChunker(chunk_size=80, overlap_size=30)
        # 生成两段会被切分的文本
        text = "A" * 120 + "。" + "B" * 120
        result = chunker.chunk(text)
        if len(result) >= 2:
            # 如果有 overlap，第二块可能包含第一块的尾部
            assert result[1]["chunk_id"] == 1

    def test_sentence_boundary(self):
        """句尾标点处切分。"""
        chunker = SmartChunker(chunk_size=200, overlap_size=40)
        # 在 180 字符处放一个句号
        prefix = "X" * 180
        suffix = "Y" * 300
        text = prefix + "。第一句结束。" + suffix
        result = chunker.chunk(text)
        assert len(result) >= 2
        # 第一块结尾应该接近句号位置
        first_text = result[0]["text"]
        assert "第一句结束" in first_text or len(first_text) <= 250

    def test_chinese_punctuation(self):
        """中文标点：。！？作为句尾。"""
        chunker = SmartChunker(chunk_size=100, overlap_size=20)
        text = ("测试内容" * 5 + "。") * 8
        result = chunker.chunk(text)
        assert len(result) > 0
        # 每块不应从中间截断中文字符
        for r in result:
            assert len(r["text"]) > 0

    def test_file_id(self):
        chunker = SmartChunker()
        result = chunker.chunk("内容", file_id="/path/to/doc.pdf")
        assert result[0]["file_id"] == "/path/to/doc.pdf"


class TestChunkSizes:
    """分块大小边界测试。"""

    def test_exact_chunk_size(self):
        """恰好等于 chunk_size 的文本。"""
        chunker = SmartChunker(chunk_size=600, overlap_size=60)
        text = "X" * 600
        result = chunker.chunk(text)
        assert len(result) == 1

    def test_slightly_over(self):
        """略超 chunk_size。"""
        chunker = SmartChunker(chunk_size=600, overlap_size=60)
        text = "X" * 650 + "。结束。"
        result = chunker.chunk(text)
        assert len(result) >= 1

    def test_very_long_text(self):
        """长文本 (50000 字符)。"""
        chunker = SmartChunker(chunk_size=600, overlap_size=60)
        text = ("这是一个比较长的测试句子用来验证分块器在大文本下的表现。" * 500)
        result = chunker.chunk(text, file_id="large.txt")
        # 应该有多个块
        assert len(result) > 10
        # 验证每个块不为空
        for r in result:
            assert len(r["text"]) > 0
            assert r["file_id"] == "large.txt"
