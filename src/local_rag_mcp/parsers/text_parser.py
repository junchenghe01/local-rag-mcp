"""纯文本/Markdown/HTML 通用解析器。

特性:
    - MD: 保留标题层级(#), 移除代码块标记
    - TXT: 按段落拆分
    - HTML: 简单标签剥离，保留文本内容
"""

import re
from pathlib import Path

from ..logger import get_logger

log = get_logger(__name__)

# HTML 标签正则
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# Markdown 代码块
_MD_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")


class TextParser:
    """通用文本解析器，支持 MD/TXT/HTML 格式。"""

    def parse(self, file_path: str) -> list[dict]:
        """解析文本文件。

        Args:
            file_path: 文件路径

        Returns:
            [{"text": "...", "source": "file_path", "section": "para_N"}, ...]
        """
        path = Path(file_path)
        if not path.exists():
            log.error("[TextParser] 文件不存在: {}", file_path)
            return []

        ext = path.suffix.lower()
        log.info("[TextParser] 解析: {} ({})", path.name, ext)

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log.exception("[TextParser] 读取失败: {}", file_path)
            return []

        if ext == ".html" or ext == ".htm":
            text = self._strip_html(raw)
        else:
            text = raw

        # 按段落拆分
        paragraphs = self._split_paragraphs(text)

        sections = []
        for i, para in enumerate(paragraphs, 1):
            stripped = para.strip()
            if not stripped:
                continue
            # 跳过过短的内容（单字符或纯标点）
            if len(stripped) < 3:
                continue
            sections.append({
                "text": stripped,
                "source": str(path.resolve()),
                "section": f"para_{i}",
            })

        log.info("[TextParser] 解析完成: {} 段落", len(sections))
        return sections

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _strip_html(self, html: str) -> str:
        """去除 HTML 标签，保留文本内容。"""
        # 移除 script/style 标签及其内容
        text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        # 替换常见块级标签为换行
        text = re.sub(r"</?(div|p|h\d|li|tr|br)[^>]*>", "\n", text, flags=re.IGNORECASE)
        # 移除所有 HTML 标签
        text = _HTML_TAG_PATTERN.sub("", text)
        # 解码常见 HTML 实体
        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
        text = text.replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
        # 压缩空白
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text

    def _split_paragraphs(self, text: str) -> list[str]:
        """按段落拆分文本。"""
        # 标准化换行
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 按双换行拆分
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]
