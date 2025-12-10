"""
工具函数测试
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import logging
from pathlib import Path

from agents.utils import (
    setup_logging, validate_data, ensure_directory, save_results, load_results,
    get_timestamp, suppress_warnings, set_random_seed, calculate_memory_usage,
    format_file_size, create_progress_callback, validate_geochemical_data,
    create_element_combinations, calculate_correlation_matrix, get_top_correlated_elements
)


class TestLogging:
    """日志功能测试类"""
    
    def test_setup_logging_default(self):
        """测试默认日志设置"""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "gold_seeker"
        assert logger.level == logging.INFO
    
    def test_setup_logging_with_file(self, tmp_path):
        """测试带文件输出的日志设置"""
        log_file = tmp_path / "test.log"
        logger = setup_logging(
            level="DEBUG",
            log_file=log_file,
            console_output=False
        )
        
        # 测试日志记录
        logger.info("Test message")
        
        # 验证文件存在
        assert log_file.exists()
        
        # 验证日志内容
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_setup_logging_no_duplicate_handlers(self):
        """测试避免重复添加处理器"""
        logger = setup_logging()
        initial_count = len(logger.handlers)
        
        # 再次调用应该不添加新的处理器
        logger2 = setup_logging()
        assert len(logger2.handlers) == initial_count


class TestDataValidation:
    """数据验证测试类"""
    
    def test_validate_data_valid(self):
        """测试有效数据验证"""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        valid, errors = validate_data(data)
        assert valid is True
        assert len(errors) == 0
    
    def test_validate_data_empty(self):
        """测试空数据验证"""
        data = pd.DataFrame()
        
        valid, errors = validate_data(data)
        assert valid is False
        assert "数据为空" in errors
    
    def test_validate_data_insufficient_samples(self):
        """测试样本数量不足"""
        data = pd.DataFrame({'A': [1, 2]})
        
        valid, errors = validate_data(data, min_samples=5)
        assert valid is False
        assert "样本数量不足" in errors[0]
    
    def test_validate_data_missing_columns(self):
        """测试缺少必需列"""
        data = pd.DataFrame({'A': [1, 2, 3]})
        
        valid, errors = validate_data(
            data, 
            required_columns=['A', 'B', 'C']
        )
        assert valid is False
        assert "缺少必需列" in errors[0]
    
    def test_validate_data_high_missing_rate(self):
        """测试高缺失率"""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, np.nan],
            'B': [1, 2, 3, 4, 5]
        })
        
        valid, errors = validate_data(data, max_missing_rate=0.4)
        assert valid is False
        assert "缺失率过高" in errors[0]
    
    def test_validate_geochemical_data(self, sample_geochemical_data, detection_limits):
        """测试地球化学数据验证"""
        elements = ['Au', 'As', 'Sb', 'Hg']
        
        valid, errors = validate_geochemical_data(
            sample_geochemical_data, elements, detection_limits
        )
        assert valid is True
        assert len(errors) == 0
    
    def test_validate_geochemical_data_missing_elements(self, sample_geochemical_data, detection_limits):
        """测试缺少元素的地球化学数据验证"""
        elements = ['Au', 'As', 'Sb', 'Hg', 'RareElement']
        
        valid, errors = validate_geochemical_data(
            sample_geochemical_data, elements, detection_limits
        )
        assert valid is False
        assert "缺少元素列" in errors[0]
    
    def test_validate_geochemical_data_missing_limits(self, sample_geochemical_data):
        """测试缺少检测限的地球化学数据验证"""
        elements = ['Au', 'As', 'Sb', 'Hg']
        incomplete_limits = {'Au': 0.05, 'As': 0.5}  # 缺少Sb, Hg
        
        valid, errors = validate_geochemical_data(
            sample_geochemical_data, elements, incomplete_limits
        )
        assert valid is False
        assert "缺少检测限" in errors[0]


class TestFileOperations:
    """文件操作测试类"""
    
    def test_ensure_directory(self, tmp_path):
        """测试确保目录存在"""
        test_dir = tmp_path / "test" / "nested" / "directory"
        
        result = ensure_directory(test_dir)
        
        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()
    
    def test_save_load_csv(self, tmp_path, sample_geochemical_data):
        """测试CSV保存和加载"""
        file_path = tmp_path / "test.csv"
        
        # 保存
        saved_path = save_results(sample_geochemical_data, file_path)
        assert saved_path.exists()
        
        # 加载
        loaded_data = load_results(saved_path)
        pd.testing.assert_frame_equal(sample_geochemical_data, loaded_data)
    
    def test_save_load_json(self, tmp_path):
        """测试JSON保存和加载"""
        test_data = {"key1": "value1", "key2": [1, 2, 3]}
        file_path = tmp_path / "test.json"
        
        # 保存
        saved_path = save_results(test_data, file_path)
        assert saved_path.exists()
        
        # 加载
        loaded_data = load_results(saved_path)
        assert test_data == loaded_data
    
    def test_save_load_pickle(self, tmp_path):
        """测试pickle保存和加载"""
        test_data = {"complex": {"nested": {"data": [1, 2, 3]}}}
        file_path = tmp_path / "test.pkl"
        
        # 保存
        saved_path = save_results(test_data, file_path)
        assert saved_path.exists()
        
        # 加载
        loaded_data = load_results(saved_path)
        assert test_data == loaded_data


class TestUtilityFunctions:
    """工具函数测试类"""
    
    def test_get_timestamp(self):
        """测试获取时间戳"""
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        assert timestamp[8] == '_'
    
    def test_suppress_warnings(self):
        """测试抑制警告"""
        import warnings
        
        # 保存原始警告状态
        original_filters = warnings.filters[:]
        
        try:
            suppress_warnings()
            
            # 验证警告被抑制
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warnings.warn("Test warning", FutureWarning)
                warnings.warn("Test warning", UserWarning)
                
                # 由于被抑制，警告列表应该为空
                assert len(w) == 0
        finally:
            # 恢复原始警告状态
            warnings.filters[:] = original_filters
    
    def test_set_random_seed(self):
        """测试设置随机种子"""
        set_random_seed(42)
        
        # 验证numpy随机性
        assert np.random.rand() == np.random.rand(0)[0]
        
        # 验证python随机性
        import random
        random.seed(42)
        assert random.random() == random.random()
    
    def test_calculate_memory_usage(self):
        """测试计算内存使用"""
        try:
            import psutil
            memory_info = calculate_memory_usage()
            
            assert isinstance(memory_info, dict)
            assert "rss_mb" in memory_info
            assert "vms_mb" in memory_info
            assert "percent" in memory_info
            
            assert memory_info["rss_mb"] > 0
            assert memory_info["vms_mb"] > 0
            assert 0 <= memory_info["percent"] <= 100
        except ImportError:
            pytest.skip("psutil not available")
    
    def test_format_file_size(self):
        """测试格式化文件大小"""
        assert format_file_size(0) == "0B"
        assert format_file_size(1024) == "1.0KB"
        assert format_file_size(1024 * 1024) == "1.0MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0GB"
    
    def test_create_progress_callback(self):
        """测试创建进度回调"""
        callback = create_progress_callback(10, "Test")
        
        # 测试回调函数
        assert callable(callback)
        
        # 测试进度更新
        callback(1)
        callback(2)
        
        # 验证进度属性
        assert hasattr(callback, 'current')
        assert callback.current == 3
    
    def test_create_element_combinations(self):
        """测试创建元素组合"""
        elements = ['Au', 'As', 'Sb', 'Hg']
        combos = create_element_combinations(elements, max_size=2)
        
        # 验证组合数量
        expected_count = 4 + 6  # C(4,1) + C(4,2)
        assert len(combos) == expected_count
        
        # 验证组合内容
        assert ['Au'] in combos
        assert ['As'] in combos
        assert ['Au', 'As'] in combos
        assert ['As', 'Sb'] in combos
    
    def test_calculate_correlation_matrix(self, sample_geochemical_data):
        """测试计算相关性矩阵"""
        corr_matrix = calculate_correlation_matrix(sample_geochemical_data)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # 方阵
        
        # 验证对角线为1
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix.values), 1.0
        )
        
        # 验证对称性
        np.testing.assert_array_almost_equal(
            corr_matrix.values, corr_matrix.T.values
        )
    
    def test_get_top_correlated_elements(self, sample_geochemical_data):
        """测试获取最相关元素"""
        corr_matrix = calculate_correlation_matrix(sample_geochemical_data)
        
        # 测试获取与Au最相关的元素
        top_elements = get_top_correlated_elements(
            corr_matrix, 'Au', top_n=3, min_correlation=0.1
        )
        
        assert isinstance(top_elements, list)
        assert len(top_elements) <= 3
        
        # 验证返回格式
        if top_elements:
            element, correlation = top_elements[0]
            assert isinstance(element, str)
            assert isinstance(correlation, (int, float))
            assert abs(correlation) >= 0.1