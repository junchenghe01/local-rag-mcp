"""RAG MCP 服务入口 --- 基于 FastMCP + LanceDB 的企业级文档检索引擎。

完整实现 MCP 五大核心特征:
    - Tools (7): ingest_file, ingest_data, query_documents, read_chunk_neighbors,
                 list_files, delete_file, status
    - Resources (2): localrag://system/status, localrag://files/list
    - Prompts (1): review-codebase
    - Sampling: 反向采样，PDF 高噪文本清洗
    - Roots: 动态工作区隔离，多项目命名空间切换

传输模式:
    - SSE (默认): HTTP + Server-Sent Events，端口 8042，守护进程模式
    - Stdio: 标准输入输出，子进程模式（调试用）

日志策略:
    - ctx.info() → 推送到 MCP 客户端
    - log.info() → 服务端 stderr 终端 + logs/server.log 文件
"""

import asyncio
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

from .logger import get_logger, TraceTimer
from .status import ServerState, server_status
from .engine import (
    init_project,
    load_project_docs,
    query_project_docs,
    get_chunk_neighbors,
    health_check as engine_health_check,
    is_supported_file,
    project_context,
)
from .watcher import FileWatcher
from .sampling import SamplingHandler

load_dotenv()

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# FastMCP 实例 (SSE 端口 8042)
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Enterprise Local RAG Knowledge Server",
    dependencies=["pydantic>=2.0"],
)

# 线程池（后台任务用）
_executor = ThreadPoolExecutor(max_workers=4)

# 关闭信号（后台任务检查此标记提前退出）
import threading
_shutdown_event = threading.Event()

# Sampling 处理器
sampling_handler = SamplingHandler()

# ---------------------------------------------------------------------------
# 服务启动日志
# ---------------------------------------------------------------------------
log.info("=" * 50)
log.info("Enterprise RAG MCP Server v2.0 启动")
log.info("Python {} | LogLevel={}", sys.version.split()[0],
         os.getenv("LOG_LEVEL", "INFO"))
log.info("Embed: {} @ {}",
         os.getenv("EMBED_MODEL_NAME", "bge-m3"),
         os.getenv("EMBED_BASE_URL", "default"))
log.info("MCP Transport: {} | Port: {}",
         os.getenv("MCP_TRANSPORT", "sse"),
         os.getenv("MCP_PORT", "8042"))
log.info("Hybrid Weight: {} | Chunk: {}/{}",
         os.getenv("RAG_HYBRID_WEIGHT", "0.7"),
         os.getenv("RAG_CHUNK_SIZE", "600"),
         os.getenv("RAG_CHUNK_OVERLAP", "60"))
log.info("=" * 50)

server_status.set(ServerState.IDLE, "就绪")


# ===================================================================
# 文件监控集成
# ===================================================================
_watcher: FileWatcher | None = None


def _on_file_event(event_type: str, file_path: str):
    """文件事件回调（在防抖线程中调用）。

    将事件转发到主事件循环执行异步处理。
    """
    log.info("[Watcher] 文件事件: {} → {}", event_type, Path(file_path).name)
    server_status.set(ServerState.INGESTING,
                      f"文件变更: {Path(file_path).name}", progress=50)

    try:
        pipeline = project_context.get("pipeline")
        store = project_context.get("store")

        if event_type == "deleted":
            # 从 LanceDB 擦除
            file_id = Path(file_path).resolve().as_posix()
            if store:
                store.delete_file(file_id)
        elif event_type in ("created", "modified"):
            # 重新索引文件
            if pipeline and is_supported_file(file_path):
                pipeline.ingest_file(file_path)

        server_status.set(ServerState.IDLE, "就绪", progress=100)
    except Exception:
        log.exception("[Watcher] 文件事件处理失败: {}", file_path)
        server_status.set_error(f"文件处理失败: {file_path}")


def _start_watcher(root_path: str):
    """启动文件监控（在后台线程中调用）。"""
    global _watcher
    if _watcher is not None:
        _watcher.stop()

    _watcher = FileWatcher(on_file_event=_on_file_event)
    _watcher.start(root_path)
    log.info("[Watcher] 文件监控线程已启动: {}", root_path)


def _stop_watcher():
    """停止文件监控。"""
    global _watcher
    if _watcher is not None:
        _watcher.stop()
        _watcher = None


# ===================================================================
# Resources (2)
# ===================================================================
@mcp.resource("localrag://system/status")
async def resource_system_status() -> str:
    """系统健康快照资源。

    Returns:
        JSON 格式的系统状态信息。
    """
    import json
    status_data = server_status.to_dict()
    store = project_context.get("store")
    if store:
        status_data["db_stats"] = store.stats()
    status_data["watcher"] = {
        "running": _watcher.is_running if _watcher else False,
        "path": _watcher.watch_path if _watcher else None,
    }
    return json.dumps(status_data, ensure_ascii=False, indent=2)


@mcp.resource("localrag://files/list")
async def resource_files_list() -> str:
    """已索引文件清单资源。

    Returns:
        JSON 格式的文件列表及统计。
    """
    import json
    store = project_context.get("store")
    if store is None:
        return json.dumps({"error": "存储未初始化"}, ensure_ascii=False)
    files = store.list_files()
    return json.dumps({"files": files, "total": len(files)}, ensure_ascii=False, indent=2)


# ===================================================================
# Prompts (1)
# ===================================================================
@mcp.prompt()
async def review_codebase(focus_area: str = "性能优化") -> str:
    """代码库全局评审模板。

    动态注入 RAG 检索 SOP，规范大模型的知识召回工作流。

    Args:
        focus_area: 审查领域 (性能优化 / 漏洞排查 / 逻辑重构)
    """
    return f"""你现在是资深的本地代码库架构师。你当前所处的项目空间已通过 Roots 挂载到本地常驻 RAG 服务器上。
针对用户要求审查的领域 **{focus_area}**，你必须严格遵循以下 SOP 操作:

1. 必须优先调用 `query_documents` 工具模糊搜索项目中包含关键逻辑的模块。
2. 如果发现召回的代码片段被切断，你必须连续调用 `read_chunk_neighbors` 补全上下文。
3. 在你的最终回答中，必须每一处代码片段都严格注明 `file_id` 和 `chunk_id` 以供用户追溯。

当前项目路径: {project_context.get('current_path', '未设置')}
"""


# ===================================================================
# Tools (7)
# ===================================================================
# ---- Tool 1: ingest_file ----
@mcp.tool()
async def ingest_file(path: str, ctx: Context) -> str:
    """索引单个文件到本地知识库。

    接收本地文件绝对路径，触发多模态解析器，完成切片、向量化并写入 LanceDB。

    Args:
        path: 本地操作系统中待建立索引的文件的绝对路径
    """
    t0 = time.time()
    await ctx.info(f"[ingest_file] 收到请求: {path}")
    log.info("[ingest_file] {}", path)

    if not is_supported_file(path):
        msg = f"不支持的文件类型: {Path(path).suffix}"
        await ctx.error(f"[ingest_file] {msg}")
        return msg

    server_status.set(ServerState.INGESTING, f"索引文件: {Path(path).name}")
    await ctx.report_progress(0, 1)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: project_context["pipeline"].ingest_file(path)
        )
        elapsed = time.time() - t0
        msg = f"索引完成 ({elapsed:.1f}s): {result['chunk_count']} 个块 → {path}"
        await ctx.info(f"[ingest_file] {msg}")
        log.info("[ingest_file] {}", msg)
        server_status.set(ServerState.IDLE, "就绪", progress=100)
    except Exception as e:
        await ctx.error(f"[ingest_file] 失败: {e}")
        log.exception("[ingest_file] 失败: {}", path)
        server_status.set_error(f"ingest_file: {e}")
        raise
    finally:
        await ctx.report_progress(1, 1)

    return msg


# ---- Tool 2: ingest_data ----
@mcp.tool()
async def ingest_data(content: str, metadata: dict | None = None, ctx: Context = None) -> str:
    """将文本/Markdown 字符串直接注入向量库。

    允许 AI 将对话中捕获的任意文本、剪贴板数据直接注入，用于构建长短期记忆。

    Args:
        content: 待注入的纯文本或 Markdown 字符串
        metadata: 可选元数据，如 {"source": "网页URL" 或 "Clipboard"}
    """
    t0 = time.time()
    await ctx.info(f"[ingest_data] 收到 {len(content)} 字符")
    log.info("[ingest_data] {} chars", len(content))

    server_status.set(ServerState.INGESTING, "注入文本数据")
    await ctx.report_progress(0, 1)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: project_context["pipeline"].ingest_text(
                content, metadata,
                source=metadata.get("source", "memory") if metadata else "memory"
            )
        )
        elapsed = time.time() - t0
        msg = f"文本注入完成 ({elapsed:.1f}s): {result['chunk_count']} 个块"
        await ctx.info(f"[ingest_data] {msg}")
        log.info("[ingest_data] {}", msg)
        server_status.set(ServerState.IDLE, "就绪", progress=100)
    except Exception as e:
        await ctx.error(f"[ingest_data] 失败: {e}")
        log.exception("[ingest_data] 失败")
        server_status.set_error(f"ingest_data: {e}")
        raise
    finally:
        await ctx.report_progress(1, 1)

    return msg


# ---- Tool 3: query_documents ----
@mcp.tool()
async def query_documents(query: str, limit: int = 4, scope: str | None = None,
                          ctx: Context = None) -> str:
    """执行混合检索（密集向量 + 稀疏全文 + RRF 重排）。

    结合 RRF 算法获取前 K 个高相关性分块，每个召回分块透传 file_id 和 chunk_id。

    Args:
        query: 提炼过的核心自然语言问题或关键词短语
        limit: 返回的高相关性切片最大数量上限 (默认 4)
        scope: 可选，指定进行过滤的文件名关键词
    """
    t0 = time.time()
    q_preview = query[:80] + "..." if len(query) > 80 else query
    await ctx.info(f"[query_documents] 收到查询: {q_preview} (limit={limit})")
    log.info("[query_documents] {} (limit={}, scope={})", q_preview, limit, scope)

    server_status.set(ServerState.QUERYING, "检索中", progress=0)
    await ctx.report_progress(0, 1)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: query_project_docs(query, limit=limit, scope=scope)
        )
        elapsed = time.time() - t0
        lines = result.split("\n") if result else []
        chunk_count = sum(1 for l in lines if l.startswith("### ["))
        await ctx.info(f"[query_documents] 完成 ({elapsed:.1f}s) | 命中 {chunk_count} 个片段")
        log.info("[query_documents] 查询完成 ({:.2f}s): {} 结果",
                 elapsed, chunk_count)
        server_status.set(ServerState.IDLE, "查询完成", progress=100)
    except Exception:
        await ctx.error(f"[query_documents] 查询异常: {q_preview}")
        log.exception("[query_documents] 查询异常: {}", q_preview)
        server_status.set_error(f"查询异常: {q_preview}")
        raise
    finally:
        await ctx.report_progress(1, 1)

    return result


# ---- Tool 4: read_chunk_neighbors ----
@mcp.tool()
async def read_chunk_neighbors(file_id: str, chunk_id: int, direction: str,
                               ctx: Context = None) -> str:
    """读取指定块的物理相邻块，防止长文本中断章取义。

    当 AI 通过 query_documents 召回的文本片段信息不全时，
    可调用此工具向上或向下捞取物理相邻的分块。

    Args:
        file_id: 目标文件的唯一标识路径
        chunk_id: 当前已知的基准分块序号
        direction: 读取邻近块的方向: "prev" | "next" | "both"
    """
    await ctx.info(f"[read_chunk_neighbors] file={file_id}, chunk={chunk_id}, dir={direction}")
    log.info("[read_chunk_neighbors] file={}, chunk={}, dir={}", file_id, chunk_id, direction)

    if direction not in ("prev", "next", "both"):
        return f"无效的 direction 参数: {direction}，请使用 prev/next/both"

    server_status.set(ServerState.QUERYING, f"读取邻近块: chunk#{chunk_id}")
    await ctx.report_progress(0, 1)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: get_chunk_neighbors(file_id, chunk_id, direction)
        )
        server_status.set(ServerState.IDLE, "就绪", progress=100)
    except Exception:
        log.exception("[read_chunk_neighbors] 失败: {}#{}", file_id, chunk_id)
        server_status.set_error(f"读取邻近块失败: {file_id}#{chunk_id}")
        raise
    finally:
        await ctx.report_progress(1, 1)

    return result


# ---- Tool 5: list_files ----
@mcp.tool()
async def list_files(ctx: Context = None) -> str:
    """列出当前向量库中已成功构建索引的所有文件列表。

    返回文件路径、切片分布总量及统计信息。
    """
    await ctx.info("[list_files] 收到请求")
    log.info("[list_files] 收到请求")

    store = project_context.get("store")
    if store is None:
        return "存储未初始化，请先设置项目路径。"

    files = store.list_files()
    if not files:
        return "当前向量库中无已索引文件。"

    import json
    lines = [f"## 已索引文件清单（共 {len(files)} 个）", ""]
    for f in files:
        lines.append(f"- **{f['file_id']}**: {f['chunk_count']} 个块")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(files, ensure_ascii=False, indent=2))
    lines.append("```")

    return "\n".join(lines)


# ---- Tool 6: delete_file ----
@mcp.tool()
async def delete_file(file_id: str, ctx: Context = None) -> str:
    """从 LanceDB 物理删除指定文件的所有 Chunk 指纹。

    连带重构 FTS 全文倒排索引，释放存储空间。

    Args:
        file_id: 要执行物理擦除的文件路径
    """
    await ctx.info(f"[delete_file] 收到请求: {file_id}")
    log.info("[delete_file] {}", file_id)

    store = project_context.get("store")
    if store is None:
        return "存储未初始化"

    server_status.set(ServerState.INDEXING, f"删除文件索引: {file_id}")
    await ctx.report_progress(0, 1)

    try:
        deleted = store.delete_file(file_id)
        msg = f"已删除 {deleted} 个块 → {file_id}"
        await ctx.info(f"[delete_file] {msg}")
        log.info("[delete_file] {}", msg)
        server_status.set(ServerState.IDLE, "就绪", progress=100)
    except Exception as e:
        await ctx.error(f"[delete_file] 失败: {e}")
        log.exception("[delete_file] 失败: {}", file_id)
        server_status.set_error(f"delete_file: {e}")
        raise
    finally:
        await ctx.report_progress(1, 1)

    return msg


# ---- Tool 7: status ----
@mcp.tool()
async def status(ctx: Context = None) -> dict:
    """系统级运行健康诊断。

    返回当前 LanceDB 表行数、磁盘空间占用、Embedding 连通性等完整状态。

    Returns:
        包含 status/phase/progress/db_stats/embedding/watcher 的健康报告。
    """
    await ctx.info("[status] 收到诊断请求")
    log.info("[status] 收到诊断请求")

    s = server_status.to_dict()

    # 数据库统计
    store = project_context.get("store")
    if store:
        s["db_stats"] = store.stats()

    # Embedding 连通性
    try:
        s["embedding"] = engine_health_check()
    except Exception as e:
        s["embedding"] = f"检查失败: {e}"

    # 文件监控状态
    s["watcher"] = {
        "running": _watcher.is_running if _watcher else False,
        "path": _watcher.watch_path if _watcher else None,
    }

    await ctx.info(f"[status] status={s.get('status')}, "
                   f"db_rows={s.get('db_stats', {}).get('row_count', 0)}")
    return s


# ===================================================================
# Roots 集成 — 动态工作区隔离
# ===================================================================
# MCP Roots 协议: 客户端 (如 Claude Code) 在初始化连接后将当前打开的
# 项目工作区路径作为 Roots 列表通知服务器。服务器从客户端获取 Roots，
# 并据此动态切换数据库命名空间和文件监控焦点。
#
# 使用 session.list_roots() 向客户端请求 Roots 列表。
# 客户端通过 notifications/roots/list_changed 通知 Roots 变更。
_current_root: str | None = None


async def _request_and_apply_roots(ctx: Context | None = None):
    """向客户端请求 Roots 列表并应用工作区隔离。

    在连接初始化和 Roots 变更通知时调用。
    """
    global _current_root
    try:
        # 尝试从 session 获取 roots
        if ctx is not None and hasattr(ctx, 'session') and ctx.session is not None:
            result = await ctx.session.list_roots()
            roots = result.roots if hasattr(result, 'roots') else []
        else:
            # 如果没有可用的 session，使用环境变量
            roots = []
    except Exception as e:
        log.debug("[Roots] 无法获取 Roots 列表: {}", e)
        return

    await _apply_roots(roots)


async def _apply_roots(roots: list):
    """应用 Roots 列表，切换工作区上下文。

    Args:
        roots: MCP Root 对象列表，每个包含 uri 和 name
    """
    global _current_root
    if not roots:
        log.debug("[Roots] Roots 列表为空，保持当前设置")
        return

    # 提取第一个 root 路径
    root_obj = roots[0]
    if hasattr(root_obj, 'uri'):
        root_uri = root_obj.uri
    elif isinstance(root_obj, dict):
        root_uri = root_obj.get("uri", "")
    else:
        root_uri = str(root_obj)

    if root_uri.startswith("file://"):
        new_root = root_uri[7:]  # 去掉 file:// 前缀
    else:
        new_root = root_uri

    # Windows 路径处理
    new_root = new_root.lstrip("/")

    if not new_root:
        return

    if new_root == _current_root:
        log.info("[Roots] 路径未变更: {}", new_root)
        return

    log.info("[Roots] 工作区切换: {} → {}", _current_root or "(none)", new_root)
    _current_root = new_root

    # 切换数据库命名空间 (通过修改 project_context 实现)
    ppath = Path(new_root).resolve()
    if ppath.exists() and ppath.is_dir():
        server_status.set(ServerState.INITIALIZING, f"切换工作区: {ppath}")
        try:
            init_project(str(ppath))
            # 切换文件监控
            if _watcher:
                _watcher.switch_root(str(ppath))
            server_status.set(ServerState.IDLE, "就绪", progress=100)
            log.info("[Roots] 工作区切换完成: {}", ppath)
        except Exception:
            log.exception("[Roots] 工作区切换失败: {}", ppath)
            server_status.set_error(f"工作区切换失败: {ppath}")
    else:
        log.warning("[Roots] 路径不存在，跳过切换: {}", new_root)


# ===================================================================
# 启动时后台加载
# ===================================================================
def _load_project_background(project_path: str):
    """后台加载项目文档索引（在独立线程中运行，不阻塞 MCP 握手）。"""
    log.info("[main] 后台加载项目: {}", project_path)
    server_status.set(ServerState.LOADING_DOCS, "启动加载项目", progress=0,
                      project_path=project_path)

    def on_progress(phase: str, pct: float):
        log.info("[main] {} ({}%%)", phase, int(pct))
        server_status.set_progress(phase, int(pct))

    try:
        result = load_project_docs(project_path, on_progress)
        log.info("[main] 项目加载完成: {}",
                 result.split("\n")[0] if result else "")

        # 启动文件监控
        _start_watcher(project_path)

        server_status.set(ServerState.IDLE, "就绪", progress=100)
    except Exception:
        log.exception("[main] 启动加载项目失败: {}", project_path)
        server_status.set_error(f"启动加载项目失败: {project_path}")
        server_status.set(ServerState.IDLE, "就绪（项目加载失败）", progress=100)


def _fix_stdio():
    """PyInstaller --noconsole 或 bootloader 可能导致 sys.stdin/stdout 为 None。

    从 OS 文件描述符重新打开，确保 stdio 通信可用。
    """
    if sys.stdin is None or sys.stdout is None:
        log.warning("[main] sys.stdin/stdout 为 None，从 OS fd 重建")
    try:
        if sys.stdin is None:
            sys.stdin = open(0, "r", encoding="utf-8", errors="replace")
        elif not hasattr(sys.stdin, "buffer"):
            sys.stdin = open(0, "r", encoding="utf-8", errors="replace")
        if sys.stdout is None:
            sys.stdout = open(1, "w", encoding="utf-8", errors="replace")
        elif not hasattr(sys.stdout, "buffer"):
            sys.stdout = open(1, "w", encoding="utf-8", errors="replace")
    except Exception:
        log.exception("[main] stdio 重建失败，尝试继续")


def main():
    """CLI 入口 --- 启动 MCP 服务 + 后台加载项目索引 + 文件监控。

    通过 MCP_TRANSPORT 环境变量切换传输模式:
        - sse (默认): HTTP + Server-Sent Events，守护进程模式
        - stdio: 标准输入输出，子进程模式
    """
    # 绑定 Sampling handler 到 MCP 实例
    sampling_handler.set_mcp_server(mcp)
    # 将 handler 注入 PdfParser（如已创建）
    try:
        from .parsers.pdf_parser import PdfParser
        PdfParser().set_noise_callback(sampling_handler.request_cleaning)
    except Exception:
        pass

    # 后台加载项目
    project_path = os.getenv("RAG_PROJECT_PATH", "").strip()
    if project_path:
        log.info("[main] 调度后台加载: {}", project_path)
        _executor.submit(_load_project_background, project_path)
    else:
        log.warning("[main] 未配置 RAG_PROJECT_PATH，服务将以空索引启动")
        server_status.set(ServerState.IDLE, "就绪（未配置项目路径）")

    # 选择传输模式
    transport = os.getenv("MCP_TRANSPORT", "sse").strip().lower()

    if transport == "sse":
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8042"))
        log.info("[main] MCP 服务以 SSE 模式启动 http://{}:{} ...",
                 mcp.settings.host, mcp.settings.port)
    else:
        _fix_stdio()
        log.info("[main] MCP 服务开始监听 stdio ...")

    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        log.info("[main] 收到 KeyboardInterrupt，服务正常退出")
    except Exception:
        log.exception("[main] 服务异常退出")
        server_status.set_error("服务异常退出")
        raise
    finally:
        log.info("[main] MCP 服务已停止")
        _shutdown_event.set()
        _stop_watcher()
        _executor.shutdown(wait=False, cancel_futures=True)
        import os as _os
        _os._exit(0)


if __name__ == "__main__":
    main()
