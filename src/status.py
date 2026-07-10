"""MCP 服务运行状态追踪 --- 线程安全的全局状态单例。

供 server.py 的业务逻辑更新状态，供 get_mcp_status 工具查询。
每次状态变更自动输出到终端日志。
"""

import enum
import threading
import time
from dataclasses import dataclass, field

from .logger import get_logger

_log = get_logger(__name__)


class ServerState(enum.Enum):
    """服务运行状态枚举."""
    INITIALIZING = "INITIALIZING"   # 服务启动中
    IDLE = "IDLE"                   # 空闲，等待请求
    LOADING_DOCS = "LOADING_DOCS"   # 正在扫描/加载文档
    INDEXING = "INDEXING"           # 正在构建向量索引
    QUERYING = "QUERYING"           # 正在执行语义查询
    INGESTING = "INGESTING"         # 正在摄取文件/文本
    WATCHING = "WATCHING"           # 文件监控运行中（空闲）
    ERROR = "ERROR"                 # 发生错误


@dataclass
class _StatusSnapshot:
    """状态快照（不可变）."""
    state: str = ServerState.INITIALIZING.value
    phase: str = "服务启动中"
    progress: int = 0
    project_path: str | None = None
    error: str | None = None
    last_updated: str = ""


# ---------------------------------------------------------------------------
# 全局状态单例（线程安全）
# ---------------------------------------------------------------------------
class ServerStatus:
    """线程安全的服务状态追踪器.

    采用粗粒度锁保护所有读写。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = ServerState.INITIALIZING
        self._phase = "服务启动中"
        self._progress = 0
        self._project_path: str | None = None
        self._error: str | None = None
        self._last_updated = 0.0

    # ---- context manager: safe state transitions ----

    def transition(self, state: ServerState, phase: str = ""):
        """返回上下文管理器，进入时设状态，退出时恢复 IDLE 或 ERROR."""
        return _StatusGuard(self, state, phase)

    # ---- atomic get / set ----

    def set(self, state: ServerState, phase: str = "",
            progress: int | None = None, project_path: str | None = None):
        with self._lock:
            prev = self._state
            self._state = state
            if phase:
                self._phase = phase
            if progress is not None:
                self._progress = progress
            if project_path is not None:
                self._project_path = project_path
            if state != ServerState.ERROR:
                self._error = None
            self._last_updated = time.time()

        # 状态变更时输出终端日志（锁外执行）
        if state != prev:
            _log.info("[状态] {} → {} | {}", prev.value, state.value,
                      phase or self._phase)

    def set_error(self, error_msg: str):
        with self._lock:
            self._state = ServerState.ERROR
            self._error = error_msg
            self._last_updated = time.time()
        _log.error("[状态] ERROR: {}", error_msg)

    def set_progress(self, phase: str, progress: int):
        """由 on_progress 回调从后台线程调用."""
        with self._lock:
            prev = self._state
            self._phase = phase
            self._progress = progress
            self._last_updated = time.time()
            # 根据 phase 关键词自动推断状态
            if not self._state == ServerState.ERROR:
                if "扫描" in phase or "加载" in phase:
                    self._state = ServerState.LOADING_DOCS
                elif "向量化" in phase or "切片" in phase or "解析" in phase or "索引" in phase or "持久化" in phase or "分块" in phase:
                    self._state = ServerState.INDEXING
                elif "摄取" in phase or "注入" in phase or "写入" in phase:
                    self._state = ServerState.INGESTING
                elif "监控" in phase or "变更" in phase:
                    self._state = ServerState.WATCHING

        # 终端日志（锁外执行，保持线程安全）
        if self._state != prev:
            _log.info("[状态] {} → {} | {} ({}%)", prev.value,
                      self._state.value, phase, progress)
        else:
            _log.info("[进度] {} | {} ({}%)", self._state.value, phase, progress)

    def snapshot(self) -> _StatusSnapshot:
        """获取当前状态快照（线程安全读）."""
        with self._lock:
            ts = time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(self._last_updated) if self._last_updated else time.localtime()
            )
            return _StatusSnapshot(
                state=self._state.value,
                phase=self._phase,
                progress=self._progress,
                project_path=self._project_path,
                error=self._error,
                last_updated=ts,
            )

    def to_dict(self) -> dict:
        s = self.snapshot()
        d = {
            "status": s.state,
            "phase": s.phase,
            "progress": s.progress,
            "project_path": s.project_path,
            "last_updated": s.last_updated,
        }
        if s.error:
            d["error"] = s.error
        return d


class _StatusGuard:
    """上下文管理器: 进入时设指定状态，退出时恢复 IDLE 或捕获异常设为 ERROR."""

    def __init__(self, status: ServerStatus, state: ServerState, phase: str):
        self._status = status
        self._state = state
        self._phase = phase

    def __enter__(self):
        self._status.set(self._state, self._phase, progress=0)

    def __exit__(self, exc_type, exc_val, _exc_tb):
        if exc_type is not None:
            self._status.set_error(
                f"{exc_type.__name__}: {exc_val}" if exc_val else exc_type.__name__
            )
        else:
            self._status.set(ServerState.IDLE, "就绪", progress=100)
        return False  # 不吞异常


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------
server_status = ServerStatus()
