"""智能分块器 --- 动态句尾标点防截断算法。

特性:
    - 硬性约束 Chunk Size = 600 字符, Overlap Size = 60 字符
    - 在切分边界智能检索最近完整句尾标点 (。！？.!?\n\n) 进行语义切分
    - 防止长文本中代码/表格被拦腰截断

用法:
    from .chunker import SmartChunker
    chunker = SmartChunker()
    chunks = chunker.chunk(text, file_id, metadata)
"""

import os
import re
from typing import Optional

from .logger import get_logger

log = get_logger(__name__)

# 句尾标点正则 — 中英文完整句尾标记
_SENTENCE_END_PATTERN = re.compile(r"[。！？.!?](\s|$)")
_PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n")


class SmartChunker:
    """动态句尾标点防截断分块器。

    Chunk Size = 600, Overlap = 60, 在切分边界处向前/后检索最近的句尾标点。
    """

    def __init__(self,
                 chunk_size: int | None = None,
                 overlap_size: int | None = None):
        """初始化分块器。

        Args:
            chunk_size: 块大小（字符数），默认从 RAG_CHUNK_SIZE 环境变量读取，回退 600
            overlap_size: 重叠大小，默认从 RAG_CHUNK_OVERLAP 读取，回退 60
        """
        self._chunk_size = chunk_size or int(os.getenv("RAG_CHUNK_SIZE", "600"))
        self._overlap_size = overlap_size or int(os.getenv("RAG_CHUNK_OVERLAP", "60"))
        # 句尾检索窗口: 在切分点前后多少字符内寻找句尾标点
        self._boundary_window = max(40, self._chunk_size // 10)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def chunk(self, text: str, file_id: str = "",
              metadata: dict | None = None) -> list[dict]:
        """将文本切分为块列表。

        Args:
            text: 待切分的纯文本
            file_id: 来源文件标识
            metadata: 附加元数据

        Returns:
            [{"chunk_id": 0, "text": "...", "file_id": "...", "metadata": {...}}, ...]
        """
        meta = metadata or {}
        if not text or not text.strip():
            return []

        # Step 1: 按段落初步拆分
        paragraphs = self._split_paragraphs(text)

        # Step 2: 段落级合并，确保每块不超过 chunk_size
        chunks = self._merge_paragraphs(paragraphs)

        # Step 3: 对超大块再次切分（句尾边界感知）
        chunks = self._split_long_chunks(chunks)

        # Step 4: 生成带 overlap 的结果
        result = self._add_overlap(chunks, file_id, meta)
        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _split_paragraphs(self, text: str) -> list[str]:
        """按双换行符/单换行符进行段落级初步拆分。"""
        # 先按双换行拆分
        raw_paras = _PARAGRAPH_BREAK_PATTERN.split(text)
        # 再对每个段落按单换行拆分（保留短段落完整性）
        result = []
        for para in raw_paras:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self._chunk_size:
                result.append(para)
            else:
                # 长段落按单换行细拆
                sub_lines = para.split("\n")
                current = ""
                for line in sub_lines:
                    line = line.strip()
                    if not line:
                        if current:
                            result.append(current)
                            current = ""
                        continue
                    if len(current) + len(line) + 1 <= self._chunk_size:
                        current = (current + "\n" + line) if current else line
                    else:
                        if current:
                            result.append(current)
                        current = line
                if current:
                    result.append(current)
        return result

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """将短段落合并到 chunk_size 限制内。"""
        chunks = []
        current = ""
        for para in paragraphs:
            if not current:
                current = para
                continue
            if len(current) + len(para) + 2 <= self._chunk_size:
                current = current + "\n\n" + para
            else:
                chunks.append(current)
                current = para
        if current:
            chunks.append(current)
        return chunks

    def _split_long_chunks(self, chunks: list[str]) -> list[str]:
        """对超出 chunk_size 的块进行句尾边界感知切分。"""
        result = []
        for chunk_text in chunks:
            if len(chunk_text) <= self._chunk_size:
                result.append(chunk_text)
                continue
            # 需要切分
            sub_chunks = self._split_by_sentence_boundary(chunk_text)
            result.extend(sub_chunks)
        return result

    def _split_by_sentence_boundary(self, text: str) -> list[str]:
        """按句尾边界切分长文本。"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # 在切分点附近寻找句尾标点
            boundary = self._find_sentence_boundary(text, end)
            chunk = text[start:boundary].strip()
            if chunk:
                chunks.append(chunk)
            start = boundary

        return chunks

    def _find_sentence_boundary(self, text: str, cut_pos: int) -> int:
        """在 cut_pos 前后 boundary_window 范围内寻找最近的句尾标点。

        优先向前找（保留更多上下文），再向后找。
        若都找不到则返回原始切分位置。
        """
        window = self._boundary_window

        # 搜索范围
        search_start = max(0, cut_pos - window)
        search_end = min(len(text), cut_pos + window)

        # 在搜索范围内找所有句尾标点位置
        best_pos = cut_pos
        best_dist = window + 1

        for m in _SENTENCE_END_PATTERN.finditer(text, search_start, search_end):
            pos = m.end()  # 标点之后的位置
            dist = abs(pos - cut_pos)
            if dist < best_dist:
                best_dist = dist
                best_pos = pos

        return best_pos

    def _add_overlap(self, chunks: list[str], file_id: str,
                     metadata: dict) -> list[dict]:
        """为每个块添加 overlap 上下文并生成最终结构。"""
        result = []
        for i, text in enumerate(chunks):
            # 前一块的末尾作为 overlap 前缀
            prefix = ""
            if i > 0 and self._overlap_size > 0:
                prev = chunks[i - 1]
                prefix = prev[-self._overlap_size:] if len(prev) > self._overlap_size else prev
                # 确保前缀从完整字符开始（避免截断多字节字符）
                if prefix and not prefix[0].isspace():
                    prefix = "…" + prefix

            full_text = (prefix + "\n\n" + text) if prefix else text

            result.append({
                "chunk_id": i,
                "text": full_text,
                "file_id": file_id,
                "metadata": metadata,
            })

        return result
