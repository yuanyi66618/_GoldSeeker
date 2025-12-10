"""
配置管理测试
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from agents.config import ConfigManager, load_config, get_config


class TestConfigManager:
    """配置管理器测试类"""
    
    def test_default_config_loading(self):
        """测试默认配置加载"""
        manager = ConfigManager()
        
        # 检查基本配置节
        assert "global" in manager.config
        assert "data" in manager.config
        assert "geochemistry" in manager.config
        assert "fractal" in manager.config
        assert "weights_of_evidence" in manager.config
        
        # 检查具体配置项
        assert manager.get("global.project_name") == "Gold-Seeker"
        assert manager.get("geochemistry.censoring_method") == "substitution"
        assert manager.get("fractal.concentration_area.n_bins") == 20
    
    def test_config_get_set(self):
        """测试配置获取和设置"""
        manager = ConfigManager()
        
        # 测试获取存在的配置
        assert manager.get("global.project_name") is not None
        assert manager.get("global.nonexistent", "default") == "default"
        
        # 测试设置配置
        manager.set("test.new_key", "test_value")
        assert manager.get("test.new_key") == "test_value"
        
        # 测试嵌套设置
        manager.set("test.nested.key", "nested_value")
        assert manager.get("test.nested.key") == "nested_value"
    
    def test_config_file_loading(self, tmp_path):
        """测试从文件加载配置"""
        # 创建临时配置文件
        config_data = {
            "global": {
                "project_name": "Test-Project",
                "version": "2.0.0"
            },
            "custom": {
                "setting": "value"
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 加载配置
        manager = ConfigManager(config_file)
        
        assert manager.get("global.project_name") == "Test-Project"
        assert manager.get("global.version") == "2.0.0"
        assert manager.get("custom.setting") == "value"
    
    def test_config_save(self, tmp_path):
        """测试配置保存"""
        manager = ConfigManager()
        
        # 修改配置
        manager.set("test.save_key", "save_value")
        
        # 保存配置
        save_file = tmp_path / "saved_config.yaml"
        manager.save(save_file)
        
        # 验证文件存在
        assert save_file.exists()
        
        # 验证内容
        with open(save_file, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["test"]["save_key"] == "save_value"
    
    def test_config_update(self):
        """测试配置更新"""
        manager = ConfigManager()
        
        original_value = manager.get("global.project_name")
        
        # 更新配置
        new_config = {
            "global": {
                "project_name": "Updated-Project",
                "new_setting": "new_value"
            }
        }
        
        manager.update(new_config)
        
        # 验证更新
        assert manager.get("global.project_name") == "Updated-Project"
        assert manager.get("global.new_setting") == "new_value"
        
        # 验证其他配置保持不变
        assert manager.get("global.version") is not None
    
    def test_helper_methods(self):
        """测试辅助方法"""
        manager = ConfigManager()
        
        # 测试检测限获取
        detection_limits = manager.get_detection_limits()
        assert isinstance(detection_limits, dict)
        assert "Au" in detection_limits
        assert "As" in detection_limits
        
        # 测试输出目录
        output_dir = manager.get_output_dir()
        assert output_dir.exists()
        assert output_dir.name == "output"
        
        # 测试缓存目录
        cache_dir = manager.get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.name == ".cache"
        
        # 测试调试模式
        debug_mode = manager.is_debug_mode()
        assert isinstance(debug_mode, bool)
        
        # 测试日志级别
        log_level = manager.get_log_level()
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        # 测试LLM配置
        llm_config = manager.get_llm_config()
        assert isinstance(llm_config, dict)
        assert "provider" in llm_config
        assert "model" in llm_config
        
        # 测试并行配置
        parallel_config = manager.get_parallel_config()
        assert isinstance(parallel_config, dict)
        assert "n_jobs" in parallel_config


class TestConfigFunctions:
    """配置函数测试类"""
    
    def test_load_config(self, test_config):
        """测试加载配置函数"""
        # 这个测试需要模拟配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            manager = load_config(config_path)
            assert isinstance(manager, ConfigManager)
            assert manager.get("global.project_name") == "Gold-Seeker-Test"
        finally:
            Path(config_path).unlink()
    
    def test_get_config(self):
        """测试获取配置函数"""
        # 第一次调用应该创建新的配置管理器
        manager1 = get_config()
        assert isinstance(manager1, ConfigManager)
        
        # 第二次调用应该返回相同的实例
        manager2 = get_config()
        assert manager1 is manager2
    
    def test_get_default_config(self):
        """测试获取默认配置函数"""
        config = get_default_config()
        assert isinstance(config, dict)
        assert "global" in config
        assert "geochemistry" in config