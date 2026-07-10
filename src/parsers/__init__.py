"""多模态 ETL 解析器包。

提供统一的文档解析接口，支持 PDF/Docx/Excel/MD/TXT/HTML 格式。
每个解析器返回 ParsedSection 列表，供分块器进一步处理。
"""

from .pdf_parser import PdfParser
from .word_parser import WordParser
from .excel_parser import ExcelParser
from .text_parser import TextParser

__all__ = ["PdfParser", "WordParser", "ExcelParser", "TextParser"]


# 解析器工厂：根据文件扩展名获取对应解析器
def get_parser(file_path: str):
    """根据文件扩展名返回对应的解析器实例。

    Args:
        file_path: 文件路径

    Returns:
        解析器实例，不支持的类型返回 None
    """
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    parsers = {
        ".pdf": PdfParser,
        ".docx": WordParser,
        ".doc": WordParser,
        ".xlsx": ExcelParser,
        ".xls": ExcelParser,
        ".md": TextParser,
        ".txt": TextParser,
        ".html": TextParser,
        ".htm": TextParser,
        ".pptx": TextParser,  # PPTX 暂用文本解析（可后续扩展）
    }
    parser_cls = parsers.get(ext)
    if parser_cls is None:
        return None
    return parser_cls()
