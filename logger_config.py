"""
统一日志配置模块
=====================
提供标准化、工业级的日志系统，支持：
  - 彩色控制台输出 (ANSI, macOS/Linux/Windows 10+)
  - 按大小轮转文件日志 (RotatingFileHandler)
  - 按天分文件错误日志 (ERROR 及以上级别)
  - 上下文感知 (RequestID / GameID / Epoch 等, 基于 contextvars 线程/协程安全)
  - 函数耗时监控装饰器 @timed
  - 异常自动堆栈记录 logger.exception()

使用方法:
  1. 在项目入口文件 (app.py / auto_update_pipeline.py 等) 最早位置调用:
        from logger_config import setup_logging
        setup_logging()

  2. 在各业务模块中:
        from logger_config import get_logger, timed, log_context, set_context
        log = get_logger(__name__)

        @timed
        def predict(uid, match_id):
            with log_context(RequestID=uid, GameID=match_id):
                log.info("开始推理")
                ...

  3. 异常处理统一使用 logger.exception() 自动附加堆栈:
        try:
            ...
        except Exception as e:
            log.exception("推理失败: %s", e)
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import datetime
import contextvars
import functools
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# =====================================================================
# Bootstrap: 确保项目根目录在 sys.path 中（兼容直接运行子目录脚本）
# =====================================================================
_BOOTSTRAP_ROOT = Path(__file__).parent.resolve()
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

# =====================================================================
# 常量与默认配置
# =====================================================================
# 项目根目录统一由 common.paths 管理（SSOT），此处作为便捷引用保留
from common.paths import PROJECT_ROOT as _PROJECT_ROOT
PROJECT_ROOT: Path = _PROJECT_ROOT
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_LEVEL = logging.INFO

CONSOLE_FORMAT = (
    "%(asctime)s [%(levelname)-8s] [%(name)s]%(context)s %(message)s"
)
FILE_FORMAT = (
    "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d]%(context)s %(message)s"
)
DATE_FMT = "%Y-%m-%d %H:%M:%S"

CONTEXT_VAR: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

_CONFIGURED = False
_ORIGINAL_LOG_RECORDS = set()


# =====================================================================
# ANSI 颜色 (控制台输出)
# =====================================================================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"


_LEVEL_COLORS = {
    logging.DEBUG: Colors.DIM + Colors.CYAN,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BG_RED + Colors.BOLD + Colors.WHITE,
}


class _ColoredFormatter(logging.Formatter):
    """控制台彩色日志格式化器"""

    def __init__(self, fmt: str = CONSOLE_FORMAT, datefmt: str = DATE_FMT,
                 use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ctx = CONTEXT_VAR.get()
        if ctx:
            ctx_parts = " ".join(f"[{k}:{v}]" for k, v in ctx.items())
            record.context = " " + ctx_parts
        else:
            record.context = ""

        msg = super().format(record)
        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno, Colors.RESET)
            msg = f"{color}{msg}{Colors.RESET}"
        return msg


class _PlainFormatter(logging.Formatter):
    """文件日志格式化器（不带颜色，带行号函数名）"""

    def __init__(self, fmt: str = FILE_FORMAT, datefmt: str = DATE_FMT):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        ctx = CONTEXT_VAR.get()
        if ctx:
            ctx_parts = " ".join(f"[{k}:{v}]" for k, v in ctx.items())
            record.context = " " + ctx_parts
        else:
            record.context = ""
        return super().format(record)


# =====================================================================
# 上下文管理
# =====================================================================
def set_context(**kwargs) -> contextvars.Token:
    """设置日志上下文 (RequestID / GameID / Epoch 等)，返回 Token 用于重置。

    示例:
        token = set_context(RequestID=req_id, Epoch=epoch)
        try:
            ...
        finally:
            CONTEXT_VAR.reset(token)
    """
    current = dict(CONTEXT_VAR.get())
    current.update(kwargs)
    return CONTEXT_VAR.set(current)


def clear_context(token: Optional[contextvars.Token] = None) -> None:
    """重置或清空日志上下文"""
    if token is not None:
        CONTEXT_VAR.reset(token)
    else:
        CONTEXT_VAR.set({})


class log_context:
    """上下文管理器，用于临时设置日志上下文。

    示例:
        with log_context(RequestID=req_id, GameID=game_id):
            log.info("处理请求中")
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        self._token = set_context(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            CONTEXT_VAR.reset(self._token)
        return False


# =====================================================================
# 耗时监控装饰器
# =====================================================================
def timed(
    _func: Optional[Callable] = None,
    *,
    level: int = logging.INFO,
    arg_names: Optional[tuple] = None,
) -> Callable:
    """函数执行耗时装饰器。

    Args:
        level: 日志级别，默认 INFO
        arg_names: 要记录的参数名元组，例如 ("match_id", "request_id")。
                   None 则不记录参数。

    示例:
        @timed
        def train_epoch():
            ...

        @timed(arg_names=("request_id",))
        def predict(request_id):
            ...
    """

    def decorator(func: Callable) -> Callable:
        is_coroutine = False
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                is_coroutine = True
        except Exception:
            pass

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            log = logging.getLogger(func.__module__)
            t0 = time.perf_counter()
            extra = ""
            if arg_names:
                try:
                    import inspect
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    parts = []
                    for name in arg_names:
                        if name in bound.arguments:
                            val = bound.arguments[name]
                            parts.append(f"{name}={val}")
                    if parts:
                        extra = " (" + ", ".join(parts) + ")"
                except Exception:
                    pass
            log.log(level, "▶ 开始 %s%s", func.__qualname__, extra)
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log.log(level, "✓ 完成 %s%s | 耗时: %.2f ms",
                        func.__qualname__, extra, elapsed_ms)
                return result
            except Exception:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log.exception("✗ 异常 %s%s | 耗时: %.2f ms",
                              func.__qualname__, extra, elapsed_ms)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            log = logging.getLogger(func.__module__)
            t0 = time.perf_counter()
            extra = ""
            if arg_names:
                try:
                    import inspect
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    parts = []
                    for name in arg_names:
                        if name in bound.arguments:
                            val = bound.arguments[name]
                            parts.append(f"{name}={val}")
                    if parts:
                        extra = " (" + ", ".join(parts) + ")"
                except Exception:
                    pass
            log.log(level, "▶ 开始 %s%s", func.__qualname__, extra)
            try:
                import asyncio
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log.log(level, "✓ 完成 %s%s | 耗时: %.2f ms",
                        func.__qualname__, extra, elapsed_ms)
                return result
            except Exception:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log.exception("✗ 异常 %s%s | 耗时: %.2f ms",
                              func.__qualname__, extra, elapsed_ms)
                raise

        return async_wrapper if is_coroutine else sync_wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


# =====================================================================
# Logger 工厂
# =====================================================================
def get_logger(name: str) -> logging.Logger:
    """获取一个 Logger 实例（推荐在各模块中使用 __name__ 作为 name）。

    如果 setup_logging() 尚未被调用，会使用一个安全的 NullHandler 配置，
    避免 "No handler found" 警告，同时不实际输出日志。
    """
    logger = logging.getLogger(name)
    if not _CONFIGURED and not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


# =====================================================================
# 全局初始化
# =====================================================================
def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = DEFAULT_LOG_LEVEL,
    console_level: int = DEFAULT_LOG_LEVEL,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    use_color: Optional[bool] = None,
    app_name: str = "app",
    enable_error_file: bool = True,
) -> logging.Logger:
    """初始化全局日志系统，在项目入口最早调用。

    Args:
        log_dir: 日志文件目录，默认 <project_root>/logs
        level: 根 Logger 的全局级别
        console_level: 控制台输出级别
        file_level: 文件输出级别
        max_bytes: 单个日志文件最大字节数（默认 10MB）
        backup_count: 轮转保留文件数（默认 5）
        use_color: 控制台是否启用彩色，None 时自动检测（TTY=启用）
        app_name: 主日志文件名前缀
        enable_error_file: 是否额外输出 ERROR 级别以上日志到独立文件

    Returns:
        根 logger
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger()

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if use_color is None:
        use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_fmt = _ColoredFormatter(use_color=use_color)
    file_fmt = _PlainFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    main_file = log_dir / f"{app_name}.log"
    file_handler = RotatingFileHandler(
        str(main_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    if enable_error_file:
        error_file = log_dir / f"{app_name}_error.log"
        error_handler = TimedRotatingFileHandler(
            str(error_file),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_fmt)
        error_handler.suffix = "%Y-%m-%d"
        root.addHandler(error_handler)

    logging.captureWarnings(True)
    _CONFIGURED = True

    root.info("=" * 70)
    root.info("日志系统初始化完成")
    root.info("日志目录: %s", log_dir)
    root.info("主日志文件: %s", main_file)
    if enable_error_file:
        root.info("错误日志: %s", error_file)
    root.info("控制台级别: %s, 文件级别: %s",
              logging.getLevelName(console_level),
              logging.getLevelName(file_level))
    root.info("=" * 70)

    return root


def get_run_logger(name: str, log_dir: Optional[Path] = None,
                   run_ts: Optional[str] = None) -> logging.Logger:
    """为单次运行（如训练任务、pipeline）创建一个带有独立日志文件的 Logger。

    该 Logger 会额外将日志写入 <log_dir>/<name>_<timestamp>.log，
    同时仍然传播到根 logger 输出到控制台和主日志。

    Args:
        name: Logger 名称（如 "train_pick"、"pipeline"）
        log_dir: 日志目录，默认 DEFAULT_LOG_DIR
        run_ts: 时间戳字符串，None 时自动生成

    Returns:
        配置好的 Logger
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_ts is None:
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"{name}_{run_ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file:
            return logger

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    logger.addHandler(fh)

    return logger


# =====================================================================
# 便捷辅助函数
# =====================================================================
def generate_request_id() -> str:
    """生成短 RequestID (8位十六进制)"""
    return uuid.uuid4().hex[:8]


def log_startup_banner(title: str, lines: Dict[str, str] = None) -> None:
    """打印启动横幅（INFO级别，适合服务入口使用）。

    Args:
        title: 横幅标题
        lines: 键值对信息，如 {"端口": "5000", "环境": "production"}
    """
    log = get_logger("startup")
    width = 60
    log.info("=" * width)
    log.info("  %s", title.center(width - 4))
    log.info("=" * width)
    if lines:
        for k, v in lines.items():
            log.info("  %s: %s", k, v)
    log.info("=" * width)


# =====================================================================
# 第三方库日志降噪
# =====================================================================
_QUIET_LOGGERS = {
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "werkzeug": logging.WARNING,
    "matplotlib": logging.WARNING,
    "PIL": logging.WARNING,
    "transformers": logging.WARNING,
    "lightgbm": logging.WARNING,
    "catboost": logging.WARNING,
    "optuna": logging.WARNING,
    "torch": logging.WARNING,
    "numpy": logging.WARNING,
    "pandas": logging.WARNING,
    "flask_cors": logging.WARNING,
    "asyncio": logging.WARNING,
}


def silence_third_party(extra: Optional[Dict[str, int]] = None) -> None:
    """抑制第三方库的冗长日志（将其级别提升到 WARNING/ERROR）。

    Args:
        extra: 额外要降噪的 logger 名 -> 级别 字典
    """
    targets = dict(_QUIET_LOGGERS)
    if extra:
        targets.update(extra)
    for name, lvl in targets.items():
        logging.getLogger(name).setLevel(lvl)
