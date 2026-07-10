"""Sampling 反向采样协作处理器。

特性:
    - 当 ETL 解析器检测到高噪文本时，反向请求前台大模型进行智能清洗
    - 通过 MCP sampling/createMessage 协议实现后台→前台的倒置控制链
    - "白嫖"前台大模型（如 Claude）的能力反哺本地知识库
    - 本地服务保持轻量（物理内存 ≤ 35MB），无需内置 LLM

工作流程:
    1. 文件解析器检测到高噪文本（noise_level > 0.35）
    2. 后台服务挂起当前写入事务
    3. 向连接的 AI 宿主发起 sampling 请求
    4. 前台大模型完成推理清洗，返回结构化文本
    5. 后台接收清洗结果，继续 ETL 流水线
"""

import asyncio
from typing import Any, Callable, Optional

from .logger import get_logger

log = get_logger(__name__)

# 高噪文本检测阈值
_NOISE_THRESHOLD = 0.35

# 采样提示词模板
_SAMPLING_PROMPT = """你是一名高效的数据清洗专家。请帮我将以下这段包含格式混乱、乱码或高噪的原始文本进行格式规整。

要求:
1. 剔除页眉、页脚、页码和无关乱码字符
2. 保留核心段落内容，修正明显的 OCR/编码错误
3. 保持原文的语言和术语不变
4. 为该段落提取出 5 个最核心的实体关键词（技术术语、名称、编号等）
5. 以标准 Markdown 格式返回，格式如下:

## 清洗后文本
[清洗后的完整文本]

## 核心关键词
- 关键词1
- 关键词2
- 关键词3
- 关键词4
- 关键词5

原始文本内容如下:
---
{raw_text}
---"""


class SamplingHandler:
    """MCP Sampling 反向采样处理器。

    封装与前台大模型的采样通信逻辑。
    """

    def __init__(self, mcp_server=None):
        """初始化采样处理器。

        Args:
            mcp_server: FastMCP 服务器实例（用于调用 request_sampling）
        """
        self._mcp = mcp_server
        self._enabled = True
        self._max_tokens = 1024

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self._max_tokens = max(256, min(value, 4096))

    def set_mcp_server(self, mcp_server):
        """绑定 MCP 服务器实例（延迟绑定，解决循环依赖）。"""
        self._mcp = mcp_server

    # ------------------------------------------------------------------
    # 采样请求
    # ------------------------------------------------------------------
    async def request_cleaning(self, raw_text: str,
                               source: str = "unknown") -> Optional[str]:
        """向前台大模型发起文本清洗请求。

        Args:
            raw_text: 高噪原始文本
            source: 文本来源（文件名等）

        Returns:
            清洗后的文本，失败则返回 None
        """
        if not self._enabled:
            log.debug("[Sampling] 已禁用，跳过清洗: {}", source)
            return None

        if self._mcp is None:
            log.warning("[Sampling] MCP 服务器未绑定，无法发起采样请求")
            return None

        if len(raw_text) < 100:
            log.debug("[Sampling] 文本过短 ({} chars)，跳过清洗", len(raw_text))
            return None

        log.info("[Sampling] 发起清洗请求: source={}, {} chars", source, len(raw_text))

        # 截断过长文本（避免超出前台模型上下文）
        truncated = raw_text[:4000] if len(raw_text) > 4000 else raw_text
        prompt = _SAMPLING_PROMPT.format(raw_text=truncated)

        try:
            result = await self._mcp.request_sampling(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
            )
            # 提取响应文本
            if hasattr(result, "content"):
                if isinstance(result.content, str):
                    cleaned = result.content
                elif isinstance(result.content, list):
                    # content 可能是 list[ContentBlock]
                    cleaned = "".join(
                        b.text if hasattr(b, "text") else str(b)
                        for b in result.content
                    )
                else:
                    cleaned = str(result.content)
            elif isinstance(result, dict):
                cleaned = result.get("content", "")
            elif isinstance(result, str):
                cleaned = result
            else:
                log.warning("[Sampling] 未知响应类型: {}", type(result))
                return None

            log.info("[Sampling] 清洗完成: source={}, {} → {} chars",
                     source, len(raw_text), len(cleaned))
            return cleaned

        except Exception as e:
            log.warning("[Sampling] 清洗请求失败 ({}): {}", source, e)
            return None

    async def request_keywords(self, text: str,
                               source: str = "unknown") -> list[str]:
        """向前台大模型请求关键词提取。

        Args:
            text: 待提取关键词的文本
            source: 来源标识

        Returns:
            关键词列表
        """
        if not self._enabled or self._mcp is None:
            return []

        prompt = (
            "请从以下文本中提取 5-10 个最核心的实体关键词"
            "（技术术语、专有名词、编号、参数名等），"
            "以 JSON 数组格式返回，不要其他内容。\n\n"
            f"文本:\n{text[:3000]}"
        )

        try:
            result = await self._mcp.request_sampling(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            import json
            content = ""
            if hasattr(result, "content"):
                content = result.content if isinstance(result.content, str) else str(result.content)
            elif isinstance(result, dict):
                content = result.get("content", "")
            else:
                content = str(result)

            # 尝试解析 JSON
            keywords = json.loads(content)
            if isinstance(keywords, list):
                return [str(k) for k in keywords[:10]]
            return []
        except Exception:
            log.debug("[Sampling] 关键词提取失败: {}", source)
            return []

    # ------------------------------------------------------------------
    # 噪声检测辅助
    # ------------------------------------------------------------------
    @staticmethod
    def should_sample(noise_level: float) -> bool:
        """判断是否需要触发采样清洗。"""
        return noise_level > _NOISE_THRESHOLD
