"""
Gold-Seeker 配置管理模块

提供配置文件加载、验证和管理功能，支持YAML格式的配置文件。
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_config.yaml"
USER_CONFIG_PATH = Path.home() / ".gold-seeker" / "config.yaml"
PROJECT_CONFIG_PATH = Path.cwd() / "config.yaml"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则按优先级查找
        """
        self.config_path = self._find_config_file(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Path:
        """查找配置文件，按优先级返回"""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            else:
                logger.warning(f"指定的配置文件不存在: {path}")
        
        # 按优先级查找配置文件
        candidates = [
            PROJECT_CONFIG_PATH,  # 项目级配置
            USER_CONFIG_PATH,    # 用户级配置
            DEFAULT_CONFIG_PATH   # 默认配置
        ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"使用配置文件: {path}")
                return path
        
        # 如果都没有找到，使用默认配置
        logger.warning("未找到配置文件，使用默认配置")
        return DEFAULT_CONFIG_PATH
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "global": {
                "project_name": "Gold-Seeker",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO",
                "random_seed": 42
            },
            "data": {
                "input_format": "csv",
                "encoding": "utf-8",
                "coordinate_system": "EPSG:4326"
            },
            "geochemistry": {
                "detection_limits": {
                    "Au": 0.05, "As": 0.5, "Sb": 0.2, "Hg": 0.01,
                    "Cu": 1.0, "Pb": 5.0, "Zn": 10.0, "Ag": 0.05
                },
                "censoring_method": "substitution",
                "transformation": {"method": "clr", "add_constant": 1e-6}
            },
            "fractal": {
                "concentration_area": {
                    "n_bins": 20,
                    "threshold_method": "knee"
                }
            },
            "weights_of_evidence": {
                "classification": {"method": "fractal", "n_classes": 2},
                "significance": {"confidence_level": 0.95}
            },
            "machine_learning": {
                "random_forest": {
                    "n_estimators": 100,
                    "random_state": 42,
                    "n_jobs": -1
                }
            },
            "visualization": {
                "style": "seaborn",
                "color_palette": "viridis",
                "figure_size": [10, 8],
                "dpi": 300
            },
            "output": {
                "output_dir": "output",
                "report_formats": ["html", "pdf", "markdown"]
            },
            "langchain": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0
                }
            },
            "performance": {
                "parallel": {"n_jobs": -1},
                "cache": {"enabled": True, "cache_dir": ".cache"}
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _validate_config(self):
        """验证配置文件"""
        required_sections = [
            "global", "data", "geochemistry", "fractal", 
            "weights_of_evidence", "machine_learning"
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"配置文件缺少必需的节: {section}")
                self.config[section] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'section.subsection.key' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持 'section.subsection.key' 格式
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"设置配置: {key} = {value}")
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        保存配置到文件
        
        Args:
            path: 保存路径，如果为None则保存到当前配置文件
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"配置已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def update(self, new_config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self._deep_update(self.config, new_config)
        logger.info("配置已更新")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_detection_limits(self) -> Dict[str, float]:
        """获取检测限配置"""
        return self.get("geochemistry.detection_limits", {})
    
    def get_output_dir(self) -> Path:
        """获取输出目录"""
        output_dir = Path(self.get("output.output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_cache_dir(self) -> Path:
        """获取缓存目录"""
        cache_dir = Path(self.get("performance.cache.cache_dir", ".cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def is_debug_mode(self) -> bool:
        """是否为调试模式"""
        return self.get("global.debug", False)
    
    def get_log_level(self) -> str:
        """获取日志级别"""
        return self.get("logging.level", "INFO")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.get("langchain.llm", {})
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """获取并行处理配置"""
        return self.get("performance.parallel", {})
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"ConfigManager(config_path={self.config_path})"


# 全局配置管理器实例
_config_manager = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> ConfigManager:
    """
    获取当前配置管理器实例
    
    Returns:
        配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    manager = ConfigManager()
    return manager.config


def reload_config(config_path: Optional[Union[str, Path]] = None):
    """
    重新加载配置
    
    Args:
        config_path: 配置文件路径
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)


def setup_config_directory():
    """设置用户配置目录"""
    config_dir = Path.home() / ".gold-seeker"
    config_dir.mkdir(exist_ok=True)
    
    # 复制默认配置文件
    user_config = config_dir / "config.yaml"
    if not user_config.exists():
        try:
            import shutil
            shutil.copy(DEFAULT_CONFIG_PATH, user_config)
            logger.info(f"已创建用户配置文件: {user_config}")
        except Exception as e:
            logger.error(f"创建用户配置文件失败: {e}")


# 初始化时设置配置目录
if not Path.home() / ".gold-seeker" / "config.yaml":
    setup_config_directory()