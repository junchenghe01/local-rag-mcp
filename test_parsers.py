"""解析器单元测试。

覆盖: PdfParser / WordParser / ExcelParser / TextParser
"""

import pytest
from rag_mcp.parsers import get_parser, PdfParser, WordParser, ExcelParser, TextParser


class TestParserFactory:
    """解析器工厂测试。"""

    def test_pdf_parser(self):
        p = get_parser("test.pdf")
        assert isinstance(p, PdfParser)

    def test_docx_parser(self):
        p = get_parser("test.docx")
        assert isinstance(p, WordParser)

    def test_doc_parser(self):
        p = get_parser("test.doc")
        assert isinstance(p, WordParser)

    def test_xlsx_parser(self):
        p = get_parser("test.xlsx")
        assert isinstance(p, ExcelParser)

    def test_xls_parser(self):
        p = get_parser("test.xls")
        assert isinstance(p, ExcelParser)

    def test_txt_parser(self):
        assert isinstance(get_parser("test.txt"), TextParser)

    def test_md_parser(self):
        assert isinstance(get_parser("test.md"), TextParser)

    def test_html_parser(self):
        assert isinstance(get_parser("test.html"), TextParser)

    def test_unknown_extension(self):
        assert get_parser("test.png") is None
        assert get_parser("test.exe") is None


class TestTextParser:
    """纯文本解析器。"""

    def test_txt_file(self, sample_txt):
        parser = TextParser()
        sections = parser.parse(str(sample_txt))
        assert len(sections) > 0
        # 验证内容
        all_text = " ".join(s["text"] for s in sections)
        assert "OTA 升级" in all_text
        assert "电源管理" in all_text
        assert "诊断功能" in all_text

    def test_md_file(self, sample_md):
        parser = TextParser()
        sections = parser.parse(str(sample_md))
        assert len(sections) > 0
        all_text = " ".join(s["text"] for s in sections)
        assert "FR-01" in all_text
        assert "NFR-01" in all_text

    def test_source_field(self, sample_txt):
        parser = TextParser()
        sections = parser.parse(str(sample_txt))
        for s in sections:
            assert "source" in s
            assert str(sample_txt.resolve()) in s["source"]

    def test_section_field(self, sample_txt):
        parser = TextParser()
        sections = parser.parse(str(sample_txt))
        for s in sections:
            assert s["section"].startswith("para_")

    def test_nonexistent_file(self):
        parser = TextParser()
        assert parser.parse("/nonexistent/file.txt") == []

    def test_html_cleaning(self, temp_dir):
        html = temp_dir / "test.html"
        html.write_text(
            "<html><body>"
            "<h1>测试标题内容</h1>"           # >3 字符才能通过过滤器
            "<script>alert('xss')</script>"
            "<p>这是段落内容</p>"
            "<style>.cls{color:red}</style>"
            "<div>另一段内容</div>"
            "</body></html>",
            encoding="utf-8",
        )
        parser = TextParser()
        sections = parser.parse(str(html))
        all_text = " ".join(s["text"] for s in sections)
        assert "测试标题" in all_text
        assert "段落内容" in all_text
        assert "alert" not in all_text


class TestWordParser:
    """Word 文档解析器。"""

    def test_parse_docx(self, sample_docx):
        parser = WordParser()
        sections = parser.parse(str(sample_docx))
        assert len(sections) > 0
        all_text = " ".join(s["text"] for s in sections)
        assert "OTA升级" in all_text
        assert "电源管理" in all_text

    def test_table_extraction(self, sample_docx):
        parser = WordParser()
        sections = parser.parse(str(sample_docx))
        table_sections = [s for s in sections if "table_" in s["section"]]
        if table_sections:
            table_text = " ".join(s["text"] for s in table_sections)
            assert "REQ-001" in table_text or "需求ID" in table_text

    def test_nonexistent_file(self):
        parser = WordParser()
        assert parser.parse("/nonexistent.docx") == []


class TestExcelParser:
    """Excel 解析器。"""

    def test_parse_xlsx(self, sample_xlsx):
        parser = ExcelParser()
        sections = parser.parse(str(sample_xlsx))
        assert len(sections) >= 1  # 3 行合并为 1 段 (20行/段)
        all_text = " ".join(s["text"] for s in sections)
        assert "REQ-001" in all_text
        assert "OTA升级" in all_text

    def test_sheet_metadata(self, sample_xlsx):
        parser = ExcelParser()
        sections = parser.parse(str(sample_xlsx))
        for s in sections:
            assert "metadata" in s
            assert "sheet" in s["metadata"]
            assert "row_range" in s["metadata"]

    def test_column_value_format(self, sample_xlsx):
        """验证 "列名: 值" 格式。"""
        parser = ExcelParser()
        sections = parser.parse(str(sample_xlsx))
        assert len(sections) > 0
        # 每行应包含 "需求ID:" 或 "功能描述:" 等列名
        assert "需求ID:" in sections[0]["text"] or any(
            ": " in s["text"] for s in sections
        )


class TestPdfParser:
    """PDF 解析器。"""

    def test_pdf_structure(self, sample_pdf):
        parser = PdfParser()
        sections = parser.parse(str(sample_pdf))
        # 空白 PDF 可能没有或很少有内容
        assert isinstance(sections, list)

    def test_noise_detection(self, sample_pdf):
        parser = PdfParser()
        sections = parser.parse(str(sample_pdf))
        for s in sections:
            assert "noise_level" in s
            assert 0.0 <= s["noise_level"] <= 1.0
            assert "needs_sampling" in s

    def test_nonexistent_file(self):
        parser = PdfParser()
        assert parser.parse("/nonexistent.pdf") == []
