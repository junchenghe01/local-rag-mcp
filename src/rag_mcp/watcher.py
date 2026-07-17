"""Watchdog 异步文件变更防抖监控引擎。

特性:
    - 实时捕获监控目录内文件的创建、修改、删除
    - 内置 1.0s 防抖状态机: 文件写入句柄释放并保持静止后才触发
    - 新增/修改 → 自动 ETL 摄取并写入 LanceDB
    - 删除 → 自动从 LanceDB 物理擦除
    - 通过 Roots 动态切换监控焦点（销毁旧监听器，绑定新路径）
"""

import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from .logger import get_logger

log = get_logger(__name__)

# 支持监控的文件扩展名
_WATCH_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".md", ".txt", ".html", ".htm", ".xlsx", ".xls"}


class _DebouncedHandler(FileSystemEventHandler):
    """带防抖的文件事件处理器。

    防抖策略:
        1. 文件事件到达后记录到 pending 字典
        2. 1.0s 内同一文件无新事件 → 触发处理
        3. 处理前验证文件是否仍然存在且可读
    """

    def __init__(self, root_path: str, callback: Callable, debounce_s: float = 1.0):
        super().__init__()
        self._root = Path(root_path)
        self._callback = callback  # async callback(event_type, file_path)
        self._debounce_s = debounce_s
        self._pending: dict[str, dict] = {}  # path -> {event, timer}
        self._lock = threading.Lock()
        self._running = True

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self._debounce("created", event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self._debounce("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self._debounce("deleted", event.src_path)

    def _debounce(self, event_type: str, file_path: str):
        """防抖处理：重置计时器，延迟触发。"""
        if not self._is_supported(file_path):
            return

        # 忽略临时文件和隐藏文件
        name = Path(file_path).name
        if name.startswith("~") or name.startswith("."):
            return

        with self._lock:
            if not self._running:
                return

            # 取消之前的定时器
            if file_path in self._pending:
                prev = self._pending[file_path]
                if prev.get("timer"):
                    prev["timer"].cancel()

            # 设置新的定时器
            timer = threading.Timer(
                self._debounce_s,
                self._fire,
                args=[event_type, file_path],
            )
            timer.daemon = True
            self._pending[file_path] = {"event_type": event_type, "timer": timer}
            timer.start()

    def _fire(self, event_type: str, file_path: str):
        """防抖定时器触发：执行实际回调。"""
        with self._lock:
            if file_path in self._pending:
                del self._pending[file_path]

        try:
            # 对于非删除事件，检查文件是否仍然存在且可读
            if event_type != "deleted":
                path = Path(file_path)
                if not path.exists():
                    log.debug("[Watcher] 文件已消失，跳过: {}", file_path)
                    return
                if not path.is_file():
                    return
                # 额外等待文件写入完成（文件大小稳定）
                if not self._wait_stable(path):
                    log.debug("[Watcher] 文件未稳定，跳过: {}", file_path)
                    return

            log.info("[Watcher] 事件触发: {} → {}", event_type, Path(file_path).name)
            self._callback(event_type, file_path)
        except Exception:
            log.exception("[Watcher] 回调异常: {}", file_path)

    def _wait_stable(self, path: Path, max_wait: float = 3.0) -> bool:
        """等待文件大小稳定（写入完成）。"""
        try:
            initial_size = path.stat().st_size
        except OSError:
            return False

        waited = 0.0
        check_interval = 0.3
        while waited < max_wait:
            time.sleep(check_interval)
            waited += check_interval
            try:
                current_size = path.stat().st_size
            except OSError:
                return False
            if current_size == initial_size:
                return True
            initial_size = current_size
        return False  # 超过最大等待时间

    def _is_supported(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in _WATCH_EXTS

    def stop(self):
        """停止所有待处理的定时器。"""
        with self._lock:
            self._running = False
            for info in self._pending.values():
                if info.get("timer"):
                    info["timer"].cancel()
            self._pending.clear()


class FileWatcher:
    """基于 Watchdog 的异步文件监控器。

    支持动态切换监控路径（Roots 隔离）。
    """

    def __init__(self, on_file_event: Callable, debounce_s: float | None = None):
        """初始化文件监控器。

        Args:
            on_file_event: async def callback(event_type: str, file_path: str)
                           event_type ∈ {"created", "modified", "deleted"}
            debounce_s: 防抖间隔（秒），默认从 RAG_WATCH_DEBOUNCE 读取
        """
        self._callback = on_file_event
        self._debounce_s = debounce_s or float(os.getenv("RAG_WATCH_DEBOUNCE", "1.0"))
        self._observer: Optional[Observer] = None
        self._handler: Optional[_DebouncedHandler] = None
        self._watch_path: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def start(self, root_path: str):
        """启动文件监控。

        Args:
            root_path: 监控根目录的绝对路径
        """
        if self._running:
            self.stop()

        path = Path(root_path).resolve()
        if not path.exists():
            log.error("[Watcher] 监控路径不存在: {}", path)
            return

        self._watch_path = str(path)
        self._handler = _DebouncedHandler(
            root_path=str(path),
            callback=self._on_event,
            debounce_s=self._debounce_s,
        )
        self._observer = Observer()
        self._observer.schedule(self._handler, str(path), recursive=True)
        self._observer.start()
        self._running = True

        log.info("[Watcher] 文件监控已启动: {} (防抖 {:.1f}s)", path, self._debounce_s)

    def stop(self):
        """停止文件监控并清理资源。"""
        if not self._running:
            return

        log.info("[Watcher] 正在停止文件监控...")
        if self._handler:
            self._handler.stop()
            self._handler = None
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        self._running = False
        self._watch_path = None
        log.info("[Watcher] 文件监控已停止")

    def switch_root(self, new_root: str):
        """切换监控根目录（Roots 隔离）。

        销毁旧监听器，将监控焦点绑定到新路径。

        Args:
            new_root: 新的监控根目录
        """
        log.info("[Watcher] 切换 Root: {} → {}", self._watch_path or "(none)", new_root)
        self.stop()
        self.start(new_root)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def watch_path(self) -> Optional[str]:
        return self._watch_path

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _on_event(self, event_type: str, file_path: str):
        """文件事件回调（在防抖线程中调用，需转发到异步上下文）。"""
        try:
            self._callback(event_type, file_path)
        except Exception:
            log.exception("[Watcher] on_event 异常: {} {}", event_type, file_path)
