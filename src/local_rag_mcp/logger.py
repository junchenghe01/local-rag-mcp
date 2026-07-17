"""全局日志模块 --- 基于 Loguru 的生产级结构化日志。

特性:
    - 彩色控制台输出（保留 ANSI 支持）
    - 按天轮转 (rotation="00:00")，自动 zip 压缩，保留 10 天
    - 全链路 Trace 耗时审计（毫秒级 Latency）
    - 线程安全（Loguru 内置）
    - 环境变量 LOG_LEVEL 动态配置（默认 INFO）
    - 全局 get_logger(__name__) 工厂函数（兼容旧接口）

用法:
    from .logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")
"""

import os
import sys
from pathlib import Path

from loguru import logger as _loguru_root

# ---------------------------------------------------------------------------
# 日志目录 & 文件
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "server_{time:YYYY-MM-DD}.log"

# ---------------------------------------------------------------------------
# 日志级别
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if _LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    _LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# 移除默认 handler，重新配置
# ---------------------------------------------------------------------------
_loguru_root.remove()

# --- 控制台输出（彩色） ---
_loguru_root.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{thread.name: <12}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=sys.stderr.isatty(),
    enqueue=True,  # 多线程安全入队
)

# --- 文件输出（按天轮转，保留 10 天，自动压缩） ---
_loguru_root.add(
    str(_LOG_FILE),
    level="DEBUG",  # 文件始终记录 DEBUG 及以上
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{thread.name: <12} | "
        "{name}:{line} | "
        "{message}"
    ),
    rotation="00:00",       # 每日凌晨切分
    retention=10,            # 保留 10 天
    compression="zip",       # 自动 zip 压缩旧日志
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# ---------------------------------------------------------------------------
# 抑制第三方库噪音
# ---------------------------------------------------------------------------
for _noisy in ("httpx", "httpcore", "urllib3", "asyncio", "watchfiles",
               "watchdog", "uvicorn", "uvicorn.access", "fastapi"):
    _loguru_root.disable(_noisy)

# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def get_logger(name: str = "local_rag_mcp"):
    """获取以 ``local_rag_mcp`` 为根的层级 logger。

    返回 Loguru logger 的 bind(name=name) 实例，兼容旧 logging API。
    支持 .info(), .debug(), .warning(), .error(), .exception() 等方法。

    Args:
        name: 推荐传 ``__name__``，自动挂载到 ``local_rag_mcp.子模块`` 命名空间。

    Returns:
        配置好的 loguru.Logger 实例（bind 了 name）。
    """
    if not name.startswith("local_rag_mcp"):
        name = f"local_rag_mcp.{name}"
    return _loguru_root.bind(name=name)


# ---------------------------------------------------------------------------
# Trace 耗时上下文管理器 (用于全链路审计)
# ---------------------------------------------------------------------------
class TraceTimer:
    """全链路 Trace 耗时审计上下文管理器。

    用法:
        with TraceTimer(log, "向量化 {n} 个节点", n=len(nodes)):
            ...  # 自动记录开始/结束 + 耗时
    """

    def __init__(self, logger, operation: str, **kwargs):
        self._log = logger
        self._op = operation.format(**kwargs) if kwargs else operation
        self._t0: float = 0.0

    def __enter__(self):
        import time
        self._t0 = time.time()
        self._log.info("[Trace] ▶ {}", self._op)
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb):
        import time
        elapsed_ms = (time.time() - self._t0) * 1000
        if exc_type is not None:
            self._log.error("[Trace] ✗ {} (FAIL after {:.0f}ms)", self._op, elapsed_ms)
        else:
            self._log.info("[Trace] ✓ {} ({:.0f}ms)", self._op, elapsed_ms)
        return False  # 不吞异常
