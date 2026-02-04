"""
日志系统
Logger Module

提供统一的日志配置和管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# 颜色代码（用于终端输出）
class Colors:
    """ANSI颜色代码"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 前景色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 背景色
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 日志级别对应的颜色
    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.BG_RED + Colors.WHITE
    }
    
    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{Colors.RESET}"
            )
        
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = 'INFO',
    console: bool = True,
    file_mode: str = 'a'
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（None表示不写文件）
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        console: 是否输出到控制台
        file_mode: 文件写入模式 ('a'追加, 'w'覆盖)
        
    Returns:
        配置好的日志记录器
        
    Example:
        >>> logger = setup_logger('my_module', log_file='logs/my_module.log')
        >>> logger.info('这是一条信息')
        >>> logger.warning('这是一条警告')
        >>> logger.error('这是一条错误')
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的handlers（避免重复）
    logger.handlers.clear()
    
    # 日志格式
    detailed_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s'
    )
    simple_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 控制台handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # 使用彩色格式化器
        console_formatter = ColoredFormatter(
            simple_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        # 创建日志目录
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        # 文件使用详细格式
        file_formatter = logging.Formatter(
            detailed_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # 防止日志传播到父logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已存在的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    日志上下文管理器
    
    用于临时改变日志级别
    
    Example:
        >>> logger = setup_logger('test')
        >>> logger.info('这会显示')
        >>> with LoggerContext(logger, 'WARNING'):
        >>>     logger.info('这不会显示')
        >>>     logger.warning('这会显示')
        >>> logger.info('这又会显示了')
    """
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


# 使用示例
if __name__ == '__main__':
    # 示例1: 基本使用
    logger = setup_logger(
        'example',
        log_file='logs/example.log',
        level='DEBUG'
    )
    
    logger.debug('这是调试信息')
    logger.info('这是普通信息')
    logger.warning('这是警告信息')
    logger.error('这是错误信息')
    logger.critical('这是严重错误信息')
    
    # 示例2: 使用上下文管理器
    print('\n使用上下文管理器:')
    with LoggerContext(logger, 'WARNING'):
        logger.info('这条不会显示')
        logger.warning('这条会显示')
    
    logger.info('这条又会显示了')
    
    # 示例3: 多个logger
    logger1 = setup_logger('module1', log_file='logs/module1.log')
    logger2 = setup_logger('module2', log_file='logs/module2.log')
    
    logger1.info('来自module1的消息')
    logger2.info('来自module2的消息')
    
    print('\n✅ 日志系统测试完成')
    print(f'日志文件已保存到: logs/')
