"""PDF 物理段落流式解析器。

特性:
    - 基于双换行符进行物理段落逻辑重构
    - 正则表达式自动过滤页眉、页脚、页码
    - 高噪检测 → 标记需要 Sampling 清洗
    - 纯文本抽取（不依赖 OCR）
"""

import re
from pathlib import Path
from typing import Callable, Optional

from pypdf import PdfReader

from ..logger import get_logger

log = get_logger(__name__)

# 页眉/页脚/页码 正则过滤模式
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                          # 纯数字行（页码）
    re.compile(r"^\s*第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$"),  # "第X页" / "第X页共Y页"
    re.compile(r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),                  # "- 1 -" 页码格式
    re.compile(r"^.{0,5}\d{1,3}\s*/\s*\d{1,3}.{0,5}$"),  # "1/10" 页码格式
]

# 高噪检测阈值
_NOISE_THRESHOLD = 0.35  # 非可读字符占比超过此值视为高噪


class PdfParser:
    """基于 PyPDF 的流式 PDF 解析器。

    按物理段落（双换行符）重构文本结构，
    自动过滤页眉页脚页码，检测高噪文本。
    """

    def __init__(self):
        self._noise_callback: Optional[Callable] = None

    def set_noise_callback(self, callback: Optional[Callable]):
        """设置高噪文本回调（用于触发 Sampling 清洗）。

        Args:
            callback: async def callback(raw_text: str) -> str
                      接收高噪原始文本，返回清洗后文本
        """
        self._noise_callback = callback

    def parse(self, file_path: str) -> list[dict]:
        """解析 PDF 文件。

        Args:
            file_path: PDF 文件路径

        Returns:
            [{"text": "...", "source": "file_path", "section": "page_N",
              "noise_level": 0.0, "needs_sampling": False}, ...]
        """
        path = Path(file_path)
        if not path.exists():
            log.error("[PdfParser] 文件不存在: {}", file_path)
            return []

        log.info("[PdfParser] 解析: {}", path.name)
        try:
            reader = PdfReader(str(path))
        except Exception:
            log.exception("[PdfParser] 无法读取 PDF: {}", file_path)
            return []

        sections = []
        for page_num, page in enumerate(reader.pages, 1):
            raw_text = page.extract_text()
            if not raw_text:
                continue

            # Step 1: 过滤页眉/页脚/页码
            cleaned = self._filter_noise_lines(raw_text)

            # Step 2: 按双换行符重构段落
            paragraphs = self._reconstruct_paragraphs(cleaned)

            # Step 3: 噪声检测
            noise_level = self._detect_noise(cleaned)

            for para in paragraphs:
                if not para.strip():
                    continue
                sections.append({
                    "text": para.strip(),
                    "source": str(path.resolve()),
                    "section": f"page_{page_num}",
                    "noise_level": noise_level,
                    "needs_sampling": noise_level > _NOISE_THRESHOLD,
                })

        log.info("[PdfParser] 解析完成: {} 段落, {} 页", len(sections), len(reader.pages))
        return sections

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _filter_noise_lines(self, text: str) -> str:
        """过滤页眉/页脚/页码行。"""
        lines = text.split("\n")
        filtered = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered.append("")
                continue
            # 检查是否匹配页眉/页脚模式
            is_noise = any(pat.match(stripped) for pat in _HEADER_FOOTER_PATTERNS)
            if not is_noise:
                filtered.append(line)
        return "\n".join(filtered)

    def _reconstruct_paragraphs(self, text: str) -> list[str]:
        """按双换行符重构物理段落。"""
        # 标准化换行
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 按双换行拆分段落
        raw = re.split(r"\n\s*\n", text)
        # 合并被单换行拆散的句子（PDF 常见问题）
        paragraphs = []
        current = ""
        for part in raw:
            part = part.strip()
            if not part:
                continue
            # 如果当前行以标点结尾，可能是完整段落
            if re.search(r"[。！？.!?]\s*$", part) and len(part) > 20:
                if current:
                    paragraphs.append(current)
                current = part
            else:
                current = (current + " " + part).strip() if current else part
        if current:
            paragraphs.append(current)
        return paragraphs

    def _detect_noise(self, text: str) -> float:
        """检测文本噪声水平（0.0-1.0）。

        基于非可读字符（控制字符、乱码）占比计算。
        """
        if not text:
            return 1.0
        # 统计非可读字符
        total = len(text)
        non_readable = sum(
            1 for c in text
            if ord(c) < 32 and c not in ("\n", "\r", "\t")  # 控制字符
            or 0xD800 <= ord(c) <= 0xDFFF                     # 代理对（常见乱码）
            or ord(c) == 0xFFFD                                 # Unicode 替换字符
        )
        return non_readable / total if total > 0 else 0.0
