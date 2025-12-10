"""
Gold-Seeker 测试配置文件

提供pytest的fixture和测试配置。
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from agents.config import ConfigManager
from agents.utils import setup_logging


@pytest.fixture(scope="session")
def test_config():
    """测试配置fixture"""
    config = {
        "global": {
            "project_name": "Gold-Seeker-Test",
            "version": "1.0.0",
            "debug": True,
            "log_level": "DEBUG",
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
                "n_estimators": 10,
                "random_state": 42,
                "n_jobs": 1
            }
        },
        "visualization": {
            "style": "seaborn",
            "color_palette": "viridis",
            "figure_size": [10, 8],
            "dpi": 100
        },
        "output": {
            "output_dir": "test_output",
            "report_formats": ["markdown"]
        },
        "langchain": {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.0
            }
        },
        "performance": {
            "parallel": {"n_jobs": 1},
            "cache": {"enabled": False}
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    return config


@pytest.fixture(scope="session")
def config_manager(test_config):
    """配置管理器fixture"""
    manager = ConfigManager()
    manager.config = test_config
    return manager


@pytest.fixture(scope="session")
def sample_geochemical_data():
    """示例地球化学数据fixture"""
    np.random.seed(42)
    n_samples = 200
    
    # 生成模拟的地球化学数据
    data = {
        'X': np.random.uniform(0, 1000, n_samples),
        'Y': np.random.uniform(0, 1000, n_samples),
        'Au': np.random.lognormal(0, 1, n_samples),
        'As': np.random.lognormal(1, 0.8, n_samples),
        'Sb': np.random.lognormal(0.5, 0.9, n_samples),
        'Hg': np.random.lognormal(-0.5, 1.2, n_samples),
        'Cu': np.random.lognormal(2, 0.7, n_samples),
        'Pb': np.random.lognormal(1.5, 0.6, n_samples),
        'Zn': np.random.lognormal(2.2, 0.5, n_samples),
        'Ag': np.random.lognormal(-0.2, 1.0, n_samples),
    }
    
    # 添加一些低于检测限的值
    detection_limits = {'Au': 0.05, 'As': 0.5, 'Sb': 0.2, 'Hg': 0.01}
    for element, limit in detection_limits.items():
        censored_mask = np.random.random(n_samples) < 0.2  # 20%的数据低于检测限
        data[element][censored_mask] = np.random.uniform(0, limit, censored_mask.sum())
    
    # 创建训练点标签
    data['Is_Deposit'] = np.zeros(n_samples, dtype=int)
    
    # 模拟矿点（高Au、As、Sb值）
    deposit_indices = np.random.choice(n_samples, size=20, replace=False)
    for idx in deposit_indices:
        data['Is_Deposit'][idx] = 1
        data['Au'][idx] *= np.random.uniform(5, 20)
        data['As'][idx] *= np.random.uniform(3, 10)
        data['Sb'][idx] *= np.random.uniform(2, 8)
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def training_points(sample_geochemical_data):
    """训练点fixture"""
    return sample_geochemical_data[sample_geochemical_data['Is_Deposit'] == 1].copy()


@pytest.fixture(scope="session")
def detection_limits():
    """检测限fixture"""
    return {
        'Au': 0.05, 'As': 0.5, 'Sb': 0.2, 'Hg': 0.01,
        'Cu': 1.0, 'Pb': 5.0, 'Zn': 10.0, 'Ag': 0.05
    }


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """输出目录fixture"""
    return tmp_path_factory.mktemp("test_output")


@pytest.fixture(autouse=True)
def setup_test_logging():
    """设置测试日志"""
    setup_logging(level="DEBUG", console_output=False)


@pytest.fixture
def mock_llm():
    """模拟LLM fixture"""
    class MockLLM:
        def __init__(self, **kwargs):
            self.model = kwargs.get('model', 'gpt-3.5-turbo')
            self.temperature = kwargs.get('temperature', 0.0)
        
        def invoke(self, prompt):
            return "Mock response for testing"
        
        def __call__(self, prompt):
            return self.invoke(prompt)
    
    return MockLLM


@pytest.fixture
def sample_r_mode_result():
    """R型聚类分析结果fixture"""
    return {
        'clusters': [
            {'elements': ['Au', 'As', 'Sb', 'Hg'], 'correlation': 0.85},
            {'elements': ['Cu', 'Pb', 'Zn'], 'correlation': 0.72},
            {'elements': ['Ag'], 'correlation': 1.0}
        ],
        'linkage_matrix': np.random.rand(10, 4),
        'dendrogram_data': {'icoord': [], 'dcoord': [], 'ivl': []}
    }


@pytest.fixture
def sample_pca_result():
    """PCA分析结果fixture"""
    return {
        'explained_variance_ratio': np.array([0.45, 0.25, 0.15, 0.10]),
        'cumulative_variance_ratio': np.array([0.45, 0.70, 0.85, 0.95]),
        'loadings': pd.DataFrame(
            np.random.randn(8, 4),
            index=['Au', 'As', 'Sb', 'Hg', 'Cu', 'Pb', 'Zn', 'Ag'],
            columns=['PC1', 'PC2', 'PC3', 'PC4']
        ),
        'scores': np.random.randn(100, 4)
    }


@pytest.fixture
def sample_fractal_result():
    """分形分析结果fixture"""
    return {
        'concentrations': np.logspace(-2, 2, 50),
        'areas': np.logspace(0, 4, 50),
        'threshold': 1.5,
        'threshold_method': 'knee',
        'slope_background': -0.5,
        'slope_anomaly': -1.5,
        'r_squared_background': 0.95,
        'r_squared_anomaly': 0.92
    }


@pytest.fixture
def sample_woe_result():
    """证据权分析结果fixture"""
    return {
        'weights': pd.DataFrame({
            'Class': ['Background', 'Anomaly'],
            'W+': [0.5, 1.2],
            'W-': [-0.3, -0.8],
            'Contrast': [0.8, 2.0],
            'Studentized_C': [1.5, 3.2],
            'Significance': [True, True]
        }),
        'confusion_matrix': np.array([[80, 10], [5, 5]]),
        'total_cells': 100,
        'deposit_cells': 10,
        'evidence_layer': np.random.randint(0, 2, 100)
    }


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        # 为没有标记的测试添加unit标记
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# 测试会话钩子
def pytest_sessionstart(session):
    """测试会话开始"""
    print("\n🧪 Gold-Seeker 测试开始")
    print("=" * 50)


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束"""
    print("=" * 50)
    print(f"🏁 测试完成，退出码: {exitstatus}")