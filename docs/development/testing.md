# Gold-Seeker 测试指南

本指南详细介绍Gold-Seeker平台的测试策略、测试方法和最佳实践。

## 📋 目录

- [测试策略](#测试策略)
- [测试环境](#测试环境)
- [单元测试](#单元测试)
- [集成测试](#集成测试)
- [端到端测试](#端到端测试)
- [性能测试](#性能测试)
- [测试数据](#测试数据)
- [持续集成](#持续集成)
- [测试报告](#测试报告)

## 🎯 测试策略

### 测试金字塔

```
        /\
       /  \
      / E2E \     <- 端到端测试 (少量)
     /______\
    /        \
   /Integration\ <- 集成测试 (适量)
  /__________\
 /            \
/   Unit Tests  \   <- 单元测试 (大量)
/______________\
```

### 测试类型

1. **单元测试**: 测试单个函数或类
2. **集成测试**: 测试组件间的交互
3. **端到端测试**: 测试完整工作流
4. **性能测试**: 测试系统性能
5. **回归测试**: 确保新功能不破坏现有功能

## 🛠️ 测试环境

### 1. 测试依赖

```bash
# 安装测试依赖
pip install -e ".[test]"

# 或安装特定测试工具
pip install pytest pytest-cov pytest-mock pytest-benchmark
pip install pytest-xdist pytest-html pytest-profiling
```

### 2. 测试配置

```python
# conftest.py
import pytest
import pandas as pd
import geopandas as gpd
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def temp_dir():
    """临时目录"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_geochemical_data():
    """示例地球化学数据"""
    return pd.DataFrame({
        'x': [1000, 1100, 1200, 1300, 1400],
        'y': [2000, 2100, 2200, 2300, 2400],
        'Au': [0.5, 1.2, 0.8, 2.1, 0.3],
        'Ag': [2.1, 3.5, 2.8, 4.2, 1.9],
        'Cu': [15.3, 18.9, 16.7, 22.1, 14.2],
        'As': [8.7, 12.4, 9.8, 15.6, 7.1],
        'Sb': [1.2, 2.1, 1.5, 3.2, 0.9],
        'Censoring': [0, 0, 0, 0, 0]
    })

@pytest.fixture
def sample_geodataframe(sample_geochemical_data):
    """示例GeoDataFrame"""
    geometry = gpd.points_from_xy(
        sample_geochemical_data['x'],
        sample_geochemical_data['y']
    )
    return gpd.GeoDataFrame(
        sample_geochemical_data,
        geometry=geometry,
        crs="EPSG:4326"
    )
```

### 3. 测试标记

```python
# pytest.ini
[tool:pytest]
markers =
    unit: 单元测试
    integration: 集成测试
    e2e: 端到端测试
    slow: 慢速测试
    gpu: 需要GPU的测试
    network: 需要网络的测试
```

## 🧪 单元测试

### 1. 测试结构

```
tests/unit/
├── test_agents/
│   ├── test_coordinator.py
│   ├── test_archivist.py
│   ├── test_spatial_analyst.py
│   ├── test_modeler.py
│   └── test_critic.py
├── test_tools/
│   ├── test_geochem_selector.py
│   ├── test_geochem_processor.py
│   ├── test_fractal_filter.py
│   └── test_woe_calculator.py
├── test_config.py
├── test_utils.py
└── test_cli.py
```

### 2. 测试示例

#### 测试工具类

```python
# tests/unit/test_tools/test_geochem_selector.py
import pytest
import numpy as np
import pandas as pd
from gold_seeker.tools import GeochemSelector

class TestGeochemSelector:
    """地球化学选择器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.selector = GeochemSelector()
        
    def test_init(self):
        """测试初始化"""
        assert self.selector is not None
        assert hasattr(self.selector, 'select_elements')
        
    def test_select_elements_basic(self, sample_geochemical_data):
        """测试基本元素选择"""
        selected = self.selector.select_elements(
            sample_geochemical_data,
            target_element='Au'
        )
        
        # 验证返回类型
        assert isinstance(selected, list)
        
        # 验证包含目标元素
        assert 'Au' in selected
        
        # 验证选择数量合理
        assert len(selected) >= 1
        assert len(selected) <= len(sample_geochemical_data.columns) - 2  # 减去x,y列
        
    def test_select_elements_invalid_target(self, sample_geochemical_data):
        """测试无效目标元素"""
        with pytest.raises(ValueError, match="目标元素不存在"):
            self.selector.select_elements(
                sample_geochemical_data,
                target_element='InvalidElement'
            )
            
    def test_select_elements_empty_data(self):
        """测试空数据"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="数据为空"):
            self.selector.select_elements(empty_data, 'Au')
            
    @pytest.mark.parametrize("method", ["r_mode_clustering", "pca", "correlation"])
    def test_select_elements_different_methods(self, sample_geochemical_data, method):
        """测试不同选择方法"""
        selected = self.selector.select_elements(
            sample_geochemical_data,
            target_element='Au',
            method=method
        )
        
        assert isinstance(selected, list)
        assert len(selected) > 0
        
    def test_perform_r_mode_analysis(self, sample_geochemical_data):
        """测试R型聚类分析"""
        elements = ['Au', 'Ag', 'Cu', 'As', 'Sb']
        result = self.selector.perform_r_mode_analysis(
            sample_geochemical_data[elements],
            elements
        )
        
        # 验证结果结构
        assert hasattr(result, 'clusters')
        assert hasattr(result, 'linkage_matrix')
        assert hasattr(result, 'dendrogram')
        
        # 验证聚类结果
        assert isinstance(result.clusters, dict)
        assert len(result.clusters) > 0
        
    def test_analyze_pca_loadings(self, sample_geochemical_data):
        """测试PCA载荷分析"""
        elements = ['Au', 'Ag', 'Cu', 'As', 'Sb']
        result = self.selector.analyze_pca_loadings(
            sample_geochemical_data[elements],
            elements
        )
        
        # 验证结果结构
        assert hasattr(result, 'loadings')
        assert hasattr(result, 'explained_variance')
        assert hasattr(result, 'components')
        
        # 验证载荷矩阵
        assert result.loadings.shape[0] == len(elements)
        assert result.loadings.shape[1] <= len(elements)
```

#### 测试代理类

```python
# tests/unit/test_agents/test_spatial_analyst.py
import pytest
from unittest.mock import Mock, patch
from gold_seeker.agents import SpatialAnalystAgent

class TestSpatialAnalystAgent:
    """空间分析代理测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.agent = SpatialAnalystAgent()
        
    def test_init(self):
        """测试初始化"""
        assert self.agent is not None
        assert hasattr(self.agent, 'analyze_geochemical_data')
        
    @patch('gold_seeker.agents.spatial_analyst.GeochemSelector')
    @patch('gold_seeker.agents.spatial_analyst.GeochemProcessor')
    @patch('gold_seeker.agents.spatial_analyst.FractalAnomalyFilter')
    def test_analyze_geochemical_data(self, mock_filter, mock_processor, mock_selector, sample_geodataframe):
        """测试地球化学数据分析"""
        # 设置mock返回值
        mock_selector.return_value.select_elements.return_value = ['Au', 'Ag', 'As']
        mock_processor.return_value.process_data.return_value = sample_geodataframe
        mock_filter.return_value.filter_anomalies.return_value = Mock(threshold=2.0)
        
        # 执行分析
        result = self.agent.analyze_geochemical_data(
            sample_geodataframe,
            target_element='Au'
        )
        
        # 验证结果
        assert result is not None
        assert hasattr(result, 'selected_elements')
        assert hasattr(result, 'processed_data')
        assert hasattr(result, 'anomalies')
        
        # 验证调用
        mock_selector.return_value.select_elements.assert_called_once()
        mock_processor.return_value.process_data.assert_called_once()
        mock_filter.return_value.filter_anomalies.assert_called_once()
```

### 3. 测试覆盖率

```python
# 运行覆盖率测试
pytest --cov=gold_seeker --cov-report=html --cov-report=term

# 查看覆盖率报告
open htmlcov/index.html
```

## 🔗 集成测试

### 1. 测试结构

```
tests/integration/
├── test_workflows.py
├── test_data_pipeline.py
├── test_agent_integration.py
└── test_tool_integration.py
```

### 2. 测试示例

#### 工作流集成测试

```python
# tests/integration/test_workflows.py
import pytest
from gold_seeker import GoldSeeker, WorkflowPlan, Task

class TestWorkflows:
    """工作流集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.gs = GoldSeeker()
        
    def test_complete_analysis_workflow(self, sample_geodataframe):
        """测试完整分析工作流"""
        # 创建工作流
        workflow = WorkflowPlan("完整分析")
        workflow.add_task(Task(
            name="元素选择",
            tool="GeochemSelector",
            parameters={"target_element": "Au"}
        ))
        workflow.add_task(Task(
            name="数据处理",
            tool="GeochemProcessor",
            parameters={"elements": ["Au", "Ag", "As"]}
        ))
        workflow.add_task(Task(
            name="异常检测",
            tool="FractalAnomalyFilter",
            parameters={"target_element": "Au"}
        ))
        
        # 执行工作流
        result = self.gs.execute_workflow(workflow, sample_geodataframe)
        
        # 验证结果
        assert result is not None
        assert result.status == "completed"
        assert len(result.results) == 3
        
    def test_agent_collaboration(self, sample_geodataframe):
        """测试代理协作"""
        # 获取代理
        coordinator = self.gs.get_agent("Coordinator")
        analyst = self.gs.get_agent("SpatialAnalyst")
        modeler = self.gs.get_agent("Modeler")
        
        # 协调代理规划任务
        workflow = coordinator.plan_task(
            "分析金矿数据",
            {"data_shape": sample_geodataframe.shape}
        )
        
        # 空间分析代理执行分析
        analysis_result = analyst.analyze_geochemical_data(
            sample_geodataframe,
            target_element="Au"
        )
        
        # 建模代理训练模型
        model_result = modeler.train_model(
            analysis_result.processed_data,
            target_element="Au",
            model_type="random_forest"
        )
        
        # 验证协作结果
        assert workflow is not None
        assert analysis_result is not None
        assert model_result is not None
```

## 🌐 端到端测试

### 1. 测试结构

```
tests/e2e/
├── test_cli_workflows.py
├── test_api_endpoints.py
├── test_real_data_scenarios.py
└── test_user_workflows.py
```

### 2. 测试示例

#### CLI工作流测试

```python
# tests/e2e/test_cli_workflows.py
import pytest
import subprocess
import tempfile
import os
from pathlib import Path

class TestCLIWorkflows:
    """CLI工作流端到端测试"""
    
    def test_analyze_command(self, sample_geochemical_data, temp_dir):
        """测试analyze命令"""
        # 保存测试数据
        data_file = temp_dir / "test_data.csv"
        sample_geochemical_data.to_csv(data_file, index=False)
        
        # 执行CLI命令
        result = subprocess.run([
            "gold-seeker", "analyze",
            "--data", str(data_file),
            "--target", "Au",
            "--output", str(temp_dir / "results")
        ], capture_output=True, text=True)
        
        # 验证执行成功
        assert result.returncode == 0
        
        # 验证输出文件
        assert (temp_dir / "results" / "analysis_results.json").exists()
        assert (temp_dir / "results" / "report.html").exists()
        
    def test_workflow_command(self, sample_geochemical_data, temp_dir):
        """测试workflow命令"""
        # 创建配置文件
        config_file = temp_dir / "config.yaml"
        config_content = """
project:
  name: "测试项目"
analysis:
  target_element: "Au"
  method: "standard"
"""
        config_file.write_text(config_content)
        
        # 保存测试数据
        data_file = temp_dir / "test_data.csv"
        sample_geochemical_data.to_csv(data_file, index=False)
        
        # 执行CLI命令
        result = subprocess.run([
            "gold-seeker", "workflow",
            "--config", str(config_file),
            "--data", str(data_file)
        ], capture_output=True, text=True)
        
        # 验证执行成功
        assert result.returncode == 0
```

## ⚡ 性能测试

### 1. 基准测试

```python
# tests/performance/test_benchmarks.py
import pytest
import numpy as np
import pandas as pd
from gold_seeker.tools import GeochemSelector, GeochemProcessor

class TestBenchmarks:
    """性能基准测试"""
    
    @pytest.mark.benchmark
    def test_selector_performance(self, benchmark):
        """测试选择器性能"""
        # 生成大数据集
        n_samples = 10000
        data = pd.DataFrame({
            f'element_{i}': np.random.lognormal(0, 1, n_samples)
            for i in range(20)
        })
        
        selector = GeochemSelector()
        
        # 基准测试
        result = benchmark(
            selector.select_elements,
            data,
            target_element='element_0'
        )
        
        assert len(result) > 0
        
    @pytest.mark.benchmark
    def test_processor_performance(self, benchmark):
        """测试处理器性能"""
        # 生成大数据集
        n_samples = 10000
        data = pd.DataFrame({
            f'element_{i}': np.random.lognormal(0, 1, n_samples)
            for i in range(10)
        })
        
        processor = GeochemProcessor()
        
        # 基准测试
        result = benchmark(
            processor.process_data,
            data,
            elements=['element_0', 'element_1', 'element_2']
        )
        
        assert result is not None
```

### 2. 内存测试

```python
# tests/performance/test_memory.py
import pytest
import psutil
import os
from gold_seeker import GoldSeeker

class TestMemory:
    """内存使用测试"""
    
    def test_memory_usage_large_dataset(self):
        """测试大数据集内存使用"""
        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 生成大数据集
        n_samples = 50000
        data = pd.DataFrame({
            'x': np.random.uniform(0, 1000000, n_samples),
            'y': np.random.uniform(0, 1000000, n_samples),
            'Au': np.random.lognormal(0, 1, n_samples),
            'Ag': np.random.lognormal(0, 1, n_samples),
            'Cu': np.random.lognormal(2, 0.5, n_samples)
        })
        
        # 执行分析
        gs = GoldSeeker()
        results = gs.quick_analyze(data, target_element="Au")
        
        # 获取最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 验证内存使用合理（不超过1GB）
        assert memory_increase < 1024, f"内存使用增加过多: {memory_increase}MB"
```

## 📊 测试数据

### 1. 数据生成

```python
# tests/fixtures/data_generator.py
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_geochemical_data(
        n_samples: int = 1000,
        elements: List[str] = None,
        spatial_extent: Dict[str, float] = None,
        correlation_structure: Dict[str, float] = None
    ) -> pd.DataFrame:
        """生成地球化学数据"""
        if elements is None:
            elements = ['Au', 'Ag', 'Cu', 'Pb', 'Zn', 'As', 'Sb']
            
        if spatial_extent is None:
            spatial_extent = {'xmin': 0, 'xmax': 10000, 'ymin': 0, 'ymax': 10000}
            
        # 生成坐标
        x = np.random.uniform(spatial_extent['xmin'], spatial_extent['xmax'], n_samples)
        y = np.random.uniform(spatial_extent['ymin'], spatial_extent['ymax'], n_samples)
        
        # 生成元素含量（对数正态分布）
        data = {'x': x, 'y': y}
        
        for element in elements:
            # 根据元素类型设置不同的分布参数
            if element in ['Au', 'Ag']:
                mean, std = 0, 1  # 贵金属
            elif element in ['Cu', 'Pb', 'Zn']:
                mean, std = 2, 0.5  # 基金属
            else:
                mean, std = 1, 0.8  # 其他元素
                
            data[element] = np.random.lognormal(mean, std, n_samples)
            
        # 添加相关性
        if correlation_structure:
            data = TestDataGenerator._add_correlation(data, correlation_structure)
            
        return pd.DataFrame(data)
    
    @staticmethod
    def _add_correlation(data: pd.DataFrame, correlation_structure: Dict[str, float]) -> pd.DataFrame:
        """添加元素间相关性"""
        # 简单的相关性实现
        for element1, element2 in correlation_structure.items():
            if element1 in data.columns and element2 in data.columns:
                correlation = correlation_structure[(element1, element2)]
                noise = np.random.normal(0, 0.1, len(data))
                data[element2] = data[element2] * (1 - correlation) + data[element1] * correlation + noise
                
        return data
    
    @staticmethod
    def generate_anomalies(
        data: pd.DataFrame,
        target_element: str,
        anomaly_percentage: float = 0.05,
        anomaly_factor: float = 5.0
    ) -> pd.DataFrame:
        """生成异常值"""
        n_anomalies = int(len(data) * anomaly_percentage)
        anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
        
        data_with_anomalies = data.copy()
        data_with_anomalies.loc[anomaly_indices, target_element] *= anomaly_factor
        
        return data_with_anomalies
```

### 2. 数据管理

```python
# tests/fixtures/__init__.py
from .data_generator import TestDataGenerator
from .real_datasets import RealDatasetLoader

# 预定义测试数据集
TEST_DATASETS = {
    'small_synthetic': {
        'n_samples': 100,
        'elements': ['Au', 'Ag', 'Cu']
    },
    'medium_synthetic': {
        'n_samples': 1000,
        'elements': ['Au', 'Ag', 'Cu', 'Pb', 'Zn', 'As', 'Sb']
    },
    'large_synthetic': {
        'n_samples': 10000,
        'elements': ['Au', 'Ag', 'Cu', 'Pb', 'Zn', 'As', 'Sb', 'Hg', 'Tl', 'Mo']
    }
}
```

## 🔄 持续集成

### 1. GitHub Actions配置

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        pytest --cov=gold_seeker --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### 2. 测试报告

```python
# tests/conftest.py
import pytest
import json
from datetime import datetime

def pytest_configure(config):
    """配置pytest钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

def pytest_html_report_title(report):
    """自定义HTML报告标题"""
    report.title = "Gold-Seeker 测试报告"

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """生成测试报告钩子"""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        # 添加测试元数据
        report.extra = getattr(report, 'extra', [])
        
        if hasattr(item, 'function'):
            # 添加测试函数信息
            report.extra.append({
                'name': '测试函数',
                'value': item.function.__name__
            })
            
            # 添加测试文档字符串
            if item.function.__doc__:
                report.extra.append({
                    'name': '测试描述',
                    'value': item.function.__doc__.strip()
                })
```

## 📈 测试报告

### 1. 覆盖率报告

```bash
# 生成HTML覆盖率报告
pytest --cov=gold_seeker --cov-report=html --cov-report=term

# 生成XML覆盖率报告（用于CI）
pytest --cov=gold_seeker --cov-report=xml
```

### 2. 性能报告

```bash
# 运行性能测试
pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json

# 生成性能报告
pytest-benchmark compare benchmark.json
```

### 3. 测试总结

```python
# scripts/generate_test_report.py
import json
import subprocess
from pathlib import Path

def generate_test_report():
    """生成测试报告"""
    # 运行测试并收集结果
    result = subprocess.run([
        "pytest", "--json-report", "--json-report-file=test_results.json"
    ], capture_output=True)
    
    # 读取测试结果
    with open("test_results.json") as f:
        test_results = json.load(f)
    
    # 生成报告
    report = {
        "summary": test_results["summary"],
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": subprocess.check_output(["python", "--version"]).decode(),
            "platform": subprocess.check_output(["python", "-c", "import platform; print(platform.platform())"]).decode()
        }
    }
    
    # 保存报告
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("测试报告已生成: test_report.json")

if __name__ == "__main__":
    generate_test_report()
```

## 🎯 最佳实践

### 1. 测试命名

```python
# 好的命名
def test_select_elements_returns_list_with_target_element():
    """测试元素选择返回包含目标元素的列表"""
    pass

def test_select_elements_raises_error_for_invalid_target():
    """测试元素选择对无效目标抛出错误"""
    pass

# 避免的命名
def test_select_elements_1():
    pass

def test_select_elements_works():
    pass
```

### 2. 测试组织

```python
# 按功能组织测试
class TestGeochemSelector:
    """地球化学选择器测试"""
    
    def test_init(self):
        """测试初始化"""
        pass
    
    def test_select_elements(self):
        """测试元素选择"""
        pass
    
    def test_edge_cases(self):
        """测试边界情况"""
        pass
```

### 3. 测试数据管理

```python
# 使用fixture管理测试数据
@pytest.fixture
def clean_data():
    """干净数据"""
    return pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
        'Au': [0.1, 0.2, 0.3]
    })

@pytest.fixture
def dirty_data():
    """脏数据"""
    return pd.DataFrame({
        'x': [1, 2, None],
        'y': [1, None, 3],
        'Au': [0.1, -0.5, 0.3]
    })
```

### 4. Mock使用

```python
# 合理使用mock
@patch('gold_seeker.tools.geochem_selector.pd.read_csv')
def test_load_data_with_mock(mock_read_csv):
    """使用mock测试数据加载"""
    mock_read_csv.return_value = pd.DataFrame({'Au': [1, 2, 3]})
    
    selector = GeochemSelector()
    result = selector.load_data("test.csv")
    
    mock_read_csv.assert_called_once_with("test.csv")
    assert len(result) == 3
```

---

通过遵循本测试指南，您可以确保Gold-Seeker平台的高质量和可靠性。测试是软件开发的重要组成部分，帮助我们构建更好的地球化学找矿预测平台。