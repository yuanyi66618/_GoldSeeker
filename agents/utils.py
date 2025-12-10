"""
Gold-Seeker 工具函数模块

提供通用的工具函数，包括日志设置、数据验证、文件操作等。
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json
import pickle

# 设置日志
def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
        console_output: 是否输出到控制台
        
    Returns:
        配置好的logger
    """
    # 创建logger
    logger = logging.getLogger("gold_seeker")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 设置格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler避免日志文件过大
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_samples: int = 10,
    max_missing_rate: float = 0.2,
    check_numeric: bool = True
) -> Tuple[bool, List[str]]:
    """
    验证数据质量
    
    Args:
        data: 要验证的数据
        required_columns: 必需的列名
        min_samples: 最小样本数
        max_missing_rate: 最大缺失率
        check_numeric: 是否检查数值类型
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查数据是否为空
    if data is None or data.empty:
        errors.append("数据为空")
        return False, errors
    
    # 检查样本数量
    if len(data) < min_samples:
        errors.append(f"样本数量不足: {len(data)} < {min_samples}")
    
    # 检查必需列
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            errors.append(f"缺少必需列: {missing_cols}")
    
    # 检查缺失率
    if max_missing_rate < 1.0:
        missing_rates = data.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > max_missing_rate].index.tolist()
        if high_missing_cols:
            errors.append(f"以下列缺失率过高: {high_missing_cols}")
    
    # 检查数值类型
    if check_numeric:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("没有数值列")
    
    return len(errors) == 0, errors


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    data: Any,
    file_path: Union[str, Path],
    format: str = "auto",
    **kwargs
) -> Path:
    """
    保存结果到文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        format: 文件格式 ('auto', 'csv', 'excel', 'json', 'pickle')
        **kwargs: 额外参数
        
    Returns:
        保存的文件路径
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    # 自动检测格式
    if format == "auto":
        format = file_path.suffix.lower().lstrip('.')
    
    try:
        if format == "csv":
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False, **kwargs)
            else:
                raise ValueError("CSV格式只支持DataFrame")
        
        elif format in ["xlsx", "excel"]:
            if isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False, **kwargs)
            else:
                raise ValueError("Excel格式只支持DataFrame")
        
        elif format == "json":
            if isinstance(data, (dict, list)):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)
            elif isinstance(data, pd.DataFrame):
                data.to_json(file_path, orient='records', indent=2, **kwargs)
            else:
                raise ValueError("JSON格式支持dict, list, DataFrame")
        
        elif format == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        logging.info(f"结果已保存到: {file_path}")
        return file_path
    
    except Exception as e:
        logging.error(f"保存文件失败: {e}")
        raise


def load_results(
    file_path: Union[str, Path],
    format: str = "auto"
) -> Any:
    """
    从文件加载数据
    
    Args:
        file_path: 文件路径
        format: 文件格式 ('auto', 'csv', 'excel', 'json', 'pickle')
        
    Returns:
        加载的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 自动检测格式
    if format == "auto":
        format = file_path.suffix.lower().lstrip('.')
    
    try:
        if format == "csv":
            return pd.read_csv(file_path)
        
        elif format in ["xlsx", "excel"]:
            return pd.read_excel(file_path)
        
        elif format == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif format == "pickle":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    except Exception as e:
        logging.error(f"加载文件失败: {e}")
        raise


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def suppress_warnings():
    """抑制常见的警告"""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保结果可重现
    
    Args:
        seed: 随机种子
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logging.info(f"随机种子已设置为: {seed}")


def calculate_memory_usage() -> Dict[str, float]:
    """
    计算内存使用情况
    
    Returns:
        内存使用信息字典
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # 物理内存
        "vms_mb": memory_info.vms / 1024 / 1024,  # 虚拟内存
        "percent": process.memory_percent(),       # 内存使用百分比
    }


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def create_progress_callback(total: int, description: str = "Processing"):
    """
    创建进度回调函数
    
    Args:
        total: 总数
        description: 描述
        
    Returns:
        进度回调函数
    """
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc=description)
        
        def callback(step: int = 1):
            pbar.update(step)
            if pbar.n >= total:
                pbar.close()
        
        return callback
    
    except ImportError:
        # 如果没有tqdm，使用简单的打印
        def callback(step: int = 1):
            current = getattr(callback, 'current', 0) + step
            callback.current = current
            if current % max(1, total // 10) == 0 or current >= total:
                print(f"{description}: {current}/{total} ({current/total*100:.1f}%)")
        
        return callback


def validate_geochemical_data(
    data: pd.DataFrame,
    elements: List[str],
    detection_limits: Dict[str, float]
) -> Tuple[bool, List[str]]:
    """
    验证地球化学数据
    
    Args:
        data: 地球化学数据
        elements: 元素列表
        detection_limits: 检测限字典
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 基本数据验证
    valid, base_errors = validate_data(data, required_columns=elements)
    if not valid:
        errors.extend(base_errors)
    
    # 检查元素是否都在数据中
    missing_elements = set(elements) - set(data.columns)
    if missing_elements:
        errors.append(f"缺少元素列: {missing_elements}")
    
    # 检查检测限
    missing_limits = set(elements) - set(detection_limits.keys())
    if missing_limits:
        errors.append(f"缺少检测限: {missing_limits}")
    
    # 检查数据范围
    for element in elements:
        if element in data.columns and element in detection_limits:
            values = data[element].dropna()
            if len(values) > 0:
                # 检查是否有负值
                if (values < 0).any():
                    errors.append(f"元素 {element} 包含负值")
                
                # 检查是否有异常高值
                max_val = values.max()
                if max_val > detection_limits[element] * 1000:
                    errors.append(f"元素 {element} 可能包含异常高值: {max_val}")
    
    return len(errors) == 0, errors


def create_element_combinations(elements: List[str], max_size: int = 3) -> List[List[str]]:
    """
    创建元素组合
    
    Args:
        elements: 元素列表
        max_size: 最大组合大小
        
    Returns:
        元素组合列表
    """
    from itertools import combinations
    
    combos = []
    for size in range(1, min(max_size + 1, len(elements) + 1)):
        for combo in combinations(elements, size):
            combos.append(list(combo))
    
    return combos


def calculate_correlation_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1
) -> pd.DataFrame:
    """
    计算相关性矩阵
    
    Args:
        data: 数据
        method: 相关性方法 ('pearson', 'kendall', 'spearman')
        min_periods: 最小样本数
        
    Returns:
        相关性矩阵
    """
    # 只选择数值列
    numeric_data = data.select_dtypes(include=[np.number])
    
    # 计算相关性
    corr_matrix = numeric_data.corr(method=method, min_periods=min_periods)
    
    return corr_matrix


def get_top_correlated_elements(
    corr_matrix: pd.DataFrame,
    target_element: str,
    top_n: int = 5,
    min_correlation: float = 0.3
) -> List[Tuple[str, float]]:
    """
    获取与目标元素最相关的元素
    
    Args:
        corr_matrix: 相关性矩阵
        target_element: 目标元素
        top_n: 返回前N个
        min_correlation: 最小相关性阈值
        
    Returns:
        (元素名, 相关系数) 列表
    """
    if target_element not in corr_matrix.columns:
        return []
    
    # 获取目标元素的相关性
    correlations = corr_matrix[target_element].drop(target_element)
    
    # 过滤低相关性
    correlations = correlations[correlations.abs() >= min_correlation]
    
    # 按绝对值排序
    correlations = correlations.sort_values(key=abs, ascending=False)
    
    # 返回前N个
    return correlations.head(top_n).items()


# 初始化时设置日志
logger = setup_logging()