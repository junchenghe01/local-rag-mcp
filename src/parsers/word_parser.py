"""Word 对象树结构化提取器。

特性:
    - 基于 python-docx 遍历段落与表格对象树
    - 保证段落层次结构清晰
    - 表格序列化为 "列A: 值A; 列B: 值B;" 格式
"""

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

from ..logger import get_logger

log = get_logger(__name__)


class WordParser:
    """基于 python-docx 的 Word 文档解析器。

    按段落和表格对象树遍历，保留文档层次结构。
    """

    def parse(self, file_path: str) -> list[dict]:
        """解析 Word 文档。

        Args:
            file_path: .docx 文件路径

        Returns:
            [{"text": "...", "source": "file_path", "section": "para_N" | "table_N"}, ...]
        """
        path = Path(file_path)
        if not path.exists():
            log.error("[WordParser] 文件不存在: {}", file_path)
            return []

        log.info("[WordParser] 解析: {}", path.name)
        try:
            doc = Document(str(path))
        except Exception:
            log.exception("[WordParser] 无法读取 Word: {}", file_path)
            return []

        sections = []
        para_idx = 0
        table_idx = 0

        # 遍历文档 body 的元素树（按出现顺序）
        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # 段落
                text = self._extract_paragraph_text(element, doc)
                if text.strip():
                    para_idx += 1
                    sections.append({
                        "text": text.strip(),
                        "source": str(path.resolve()),
                        "section": f"para_{para_idx}",
                    })

            elif tag == "tbl":
                # 表格
                table_text = self._extract_table_text(element, doc)
                if table_text.strip():
                    table_idx += 1
                    sections.append({
                        "text": table_text.strip(),
                        "source": str(path.resolve()),
                        "section": f"table_{table_idx}",
                    })

        log.info("[WordParser] 解析完成: {} 段落, {} 表格", para_idx, table_idx)
        return sections

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _extract_paragraph_text(self, p_element, doc) -> str:
        """从 XML 段落元素提取文本，保留样式信息。"""
        texts = []
        # 遍历段落内的 run 元素
        for r in p_element.iter(qn("w:r")):
            t_elements = r.findall(qn("w:t"))
            for t in t_elements:
                if t.text:
                    texts.append(t.text)

        # 处理段落内的超链接
        for hyperlink in p_element.iter(qn("w:hyperlink")):
            for r in hyperlink.iter(qn("w:r")):
                for t in r.findall(qn("w:t")):
                    if t.text:
                        texts.append(t.text)

        return "".join(texts)

    def _extract_table_text(self, tbl_element, doc) -> str:
        """从 XML 表格元素提取文本，序列化为结构化文本。

        格式:
            行1: 列A: 值A; 列B: 值B;
            行2: 列A: 值A; 列B: 值B;
        """
        rows = tbl_element.findall(qn("w:tr"))
        if not rows:
            return ""

        # 提取表头（第一行）
        header = []
        data_rows = []

        for row_idx, row in enumerate(rows):
            cells = row.findall(qn("w:tc"))
            cell_texts = []
            for cell in cells:
                # 提取单元格内所有段落文本
                cell_parts = []
                for p in cell.iter(qn("w:p")):
                    para_texts = []
                    for r in p.iter(qn("w:r")):
                        for t in r.findall(qn("w:t")):
                            if t.text:
                                para_texts.append(t.text)
                    cell_parts.append("".join(para_texts))
                cell_texts.append(" ".join(cell_parts).strip())

            if row_idx == 0:
                header = cell_texts
            else:
                data_rows.append(cell_texts)

        # 序列化
        lines = []
        if header:
            lines.append("表头: " + " | ".join(header))

        for i, row in enumerate(data_rows):
            if header and len(header) == len(row):
                # 有表头: "列名: 值;"
                pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
                lines.append(f"行{i + 1}: " + "; ".join(pairs))
            else:
                # 无表头: 直接拼接
                lines.append(f"行{i + 1}: " + "; ".join(row))

        return "\n".join(lines)
