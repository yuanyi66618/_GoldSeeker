# Gold-Seeker 贡献指南

欢迎为Gold-Seeker地球化学找矿预测智能平台做出贡献！本指南将帮助您了解如何参与项目开发。

## 📋 目录

- [贡献方式](#贡献方式)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [测试指南](#测试指南)
- [文档贡献](#文档贡献)
- [问题报告](#问题报告)
- [社区参与](#社区参与)

## 🤝 贡献方式

### 1. 代码贡献

- 修复bug
- 添加新功能
- 优化性能
- 重构代码
- 添加测试

### 2. 文档贡献

- 改进现有文档
- 添加新文档
- 翻译文档
- 添加示例
- 修正错误

### 3. 问题反馈

- 报告bug
- 提出功能请求
- 改进建议
- 使用问题

### 4. 社区参与

- 回答问题
- 分享经验
- 推广项目
- 组织活动

## 🛠️ 开发环境设置

### 1. Fork和克隆

```bash
# Fork项目到您的GitHub账户
# 然后克隆您的fork
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker

# 添加上游仓库
git remote add upstream https://github.com/original-owner/Gold-Seeker.git
```

### 2. 创建开发环境

```bash
# 创建虚拟环境
python -m venv dev-env
source dev-env/bin/activate  # Linux/Mac
# 或
dev-env\Scripts\activate  # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

### 3. 安装pre-commit钩子

```bash
# 安装pre-commit
pre-commit install

# 运行pre-commit检查
pre-commit run --all-files
```

### 4. 配置IDE

推荐使用VS Code，安装以下扩展：

- Python
- Pylance
- Black Formatter
- isort
- flake8
- mypy

## 📝 代码规范

### 1. Python代码风格

我们使用以下工具确保代码质量：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

#### 代码格式化

```bash
# 格式化代码
black .

# 排序导入
isort .

# 检查代码
flake8 .

# 类型检查
mypy .
```

#### 代码风格示例

```python
# 好的示例
from typing import List, Dict, Any, Optional
import pandas as pd
import geopandas as gpd

class GeochemicalAnalyzer:
    """地球化学分析器"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self._data: Optional[gpd.GeoDataFrame] = None
    
    def load_data(self, file_path: str) -> gpd.GeoDataFrame:
        """加载地球化学数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            GeoDataFrame: 加载的数据
            
        Raises:
            FileNotFoundError: 文件不存在
            DataFormatError: 数据格式错误
        """
        try:
            data = gpd.read_file(file_path)
            self._validate_data(data)
            self._data = data
            return data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"数据文件不存在: {file_path}") from e
    
    def _validate_data(self, data: gpd.GeoDataFrame) -> None:
        """验证数据格式
        
        Args:
            data: 待验证的数据
            
        Raises:
            DataFormatError: 数据格式错误
        """
        required_columns = ['x', 'y', 'Au']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise DataFormatError(f"缺少必需列: {missing_columns}")
```

### 2. 命名规范

#### 类名

使用PascalCase（大驼峰）：

```python
class GeochemicalProcessor:
    pass

class SpatialAnalystAgent:
    pass
```

#### 函数和变量名

使用snake_case（小写下划线）：

```python
def process_geochemical_data():
    pass

element_concentration = 1.5
```

#### 常量名

使用UPPER_CASE：

```python
DEFAULT_CONFIG_PATH = "config.yaml"
MAX_MEMORY_USAGE = "8GB"
```

#### 私有方法

使用单下划线前缀：

```python
class MyClass:
    def _private_method(self):
        pass
    
    def __special_method(self):
        pass
```

### 3. 文档字符串

使用Google风格的文档字符串：

```python
def calculate_weights(data: gpd.GeoDataFrame, 
                      target_element: str,
                      threshold: float) -> Dict[str, float]:
    """计算证据权
    
    计算指定元素的证据权重，包括正权重、负权重和对比度。
    
    Args:
        data: 地理空间数据
        target_element: 目标元素名称
        threshold: 异常阈值
        
    Returns:
        包含权重的字典:
            - w_plus: 正权重
            - w_minus: 负权重
            - contrast: 对比度
            
    Raises:
        ValueError: 当目标元素不存在时
        DataValidationError: 当数据验证失败时
        
    Examples:
        >>> data = gpd.read_file("data.shp")
        >>> weights = calculate_weights(data, "Au", 2.5)
        >>> print(weights["w_plus"])
        1.23
    """
```

### 4. 类型注解

所有公共API都应该有类型注解：

```python
from typing import List, Dict, Any, Optional, Union

def process_elements(elements: List[str], 
                     data: pd.DataFrame,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """处理元素列表"""
    pass

class Analyzer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._results: Optional[Dict[str, Any]] = None
    
    def get_results(self) -> Dict[str, Any]:
        """获取结果"""
        if self._results is None:
            return {}
        return self._results
```

## 📦 提交规范

### 1. 分支策略

- **main**: 主分支，稳定版本
- **develop**: 开发分支
- **feature/***: 功能分支
- **bugfix/***: 修复分支
- **hotfix/***: 热修复分支

#### 创建功能分支

```bash
# 切换到develop分支
git checkout develop

# 创建功能分支
git checkout -b feature/new-analysis-method

# 开发完成后
git add .
git commit -m "feat: 添加新的分析方法"
git push origin feature/new-analysis-method
```

### 2. 提交信息格式

使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### 类型说明

- **feat**: 新功能
- **fix**: 修复bug
- **docs**: 文档更新
- **style**: 代码格式化
- **refactor**: 重构
- **test**: 测试相关
- **chore**: 构建过程或辅助工具的变动

#### 示例

```bash
# 新功能
git commit -m "feat(analysis): 添加C-A分形分析方法"

# 修复bug
git commit -m "fix(data): 修复CSV文件加载时的编码问题"

# 文档更新
git commit -m "docs(readme): 更新安装说明"

# 重构
git commit -m "refactor(tools): 重构地球化学工具类结构"
```

### 3. Pull Request流程

#### 创建PR

1. 推送分支到您的fork
2. 在GitHub上创建Pull Request
3. 填写PR模板
4. 等待代码审查

#### PR模板

```markdown
## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 其他

## 变更描述
简要描述您的变更内容

## 测试
- [ ] 添加了新的测试
- [ ] 所有测试通过
- [ ] 手动测试通过

## 检查清单
- [ ] 代码符合项目规范
- [ ] 添加了必要的文档
- [ ] 更新了CHANGELOG.md
- [ ] 没有引入新的警告

## 相关Issue
Closes #123
```

## 🧪 测试指南

### 1. 测试结构

```
tests/
├── unit/           # 单元测试
├── integration/    # 集成测试
├── e2e/           # 端到端测试
├── fixtures/      # 测试数据
└── conftest.py    # pytest配置
```

### 2. 编写测试

#### 单元测试

```python
import pytest
import pandas as pd
from gold_seeker.tools import GeochemSelector

class TestGeochemSelector:
    """地球化学选择器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.selector = GeochemSelector()
        self.test_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [1, 2, 3, 4, 5],
            'Au': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Ag': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Cu': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
    
    def test_select_elements(self):
        """测试元素选择"""
        selected = self.selector.select_elements(
            self.test_data, 
            target_element='Au'
        )
        
        assert isinstance(selected, list)
        assert 'Au' in selected
        assert len(selected) > 0
    
    def test_select_elements_invalid_target(self):
        """测试无效目标元素"""
        with pytest.raises(ValueError):
            self.selector.select_elements(
                self.test_data,
                target_element='InvalidElement'
            )
    
    @pytest.mark.parametrize("method", ["r_mode_clustering", "pca", "correlation"])
    def test_select_elements_different_methods(self, method):
        """测试不同选择方法"""
        selected = self.selector.select_elements(
            self.test_data,
            target_element='Au',
            method=method
        )
        
        assert isinstance(selected, list)
        assert len(selected) > 0
```

#### 集成测试

```python
import pytest
from gold_seeker import GoldSeeker

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        gs = GoldSeeker()
        
        # 加载测试数据
        data = gs.load_data("tests/fixtures/test_data.csv")
        
        # 执行分析
        results = gs.quick_analyze(data, target_element="Au")
        
        # 验证结果
        assert results is not None
        assert hasattr(results, 'selected_elements')
        assert hasattr(results, 'anomalies')
        assert hasattr(results, 'weights')
```

### 3. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_selector.py

# 运行特定测试类
pytest tests/unit/test_selector.py::TestGeochemSelector

# 运行特定测试方法
pytest tests/unit/test_selector.py::TestGeochemSelector::test_select_elements

# 生成覆盖率报告
pytest --cov=gold_seeker --cov-report=html

# 运行性能测试
pytest tests/performance/ --benchmark-only
```

### 4. 测试数据

使用pytest fixtures管理测试数据：

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_geochemical_data():
    """示例地球化学数据"""
    return pd.DataFrame({
        'x': [1000, 1100, 1200, 1300, 1400],
        'y': [2000, 2100, 2200, 2300, 2400],
        'Au': [0.5, 1.2, 0.8, 2.1, 0.3],
        'Ag': [2.1, 3.5, 2.8, 4.2, 1.9],
        'Cu': [15.3, 18.9, 16.7, 22.1, 14.2],
        'Censoring': [0, 0, 0, 0, 0]
    })

@pytest.fixture
def config_dict():
    """示例配置字典"""
    return {
        "data": {
            "coordinate_system": "EPSG:4326",
            "format": "csv"
        },
        "analysis": {
            "target_element": "Au",
            "method": "standard"
        }
    }
```

## 📚 文档贡献

### 1. 文档结构

```
docs/
├── user_guide/     # 用户指南
├── development/    # 开发文档
├── theory/         # 理论基础
├── examples/       # 示例
└── reference/      # 参考资料
```

### 2. 文档格式

使用Markdown格式，支持：

- 代码块
- 表格
- 链接
- 图片
- 数学公式

#### 代码示例

```python
# 示例：加载和分析数据
from gold_seeker import GoldSeeker

# 初始化平台
gs = GoldSeeker()

# 加载数据
data = gs.load_data("geochemical_data.csv")

# 执行分析
results = gs.quick_analyze(data, target_element="Au")

# 查看结果
print(results.summary())
```

#### 数学公式

使用LaTeX格式：

```markdown
C-A分形模型：

$$N(C) = F \cdot C^{-D}$$

其中：
- $N(C)$ 是含量大于$C$的样本数
- $F$ 是常数
- $D$ 是分形维数
```

### 3. 文档审查

- 检查语法和拼写
- 验证代码示例
- 确保链接有效
- 检查格式一致性

## 🐛 问题报告

### 1. 报告bug

使用GitHub Issues报告bug，包含：

- 问题描述
- 重现步骤
- 期望行为
- 实际行为
- 环境信息
- 相关日志

#### Bug报告模板

```markdown
## Bug描述
简要描述bug

## 重现步骤
1. 执行命令...
2. 点击...
3. 查看错误

## 期望行为
描述您期望发生的情况

## 实际行为
描述实际发生的情况

## 环境信息
- OS: [e.g. Windows 10, macOS 11.0, Ubuntu 20.04]
- Python版本: [e.g. 3.9.0]
- Gold-Seeker版本: [e.g. 1.0.0]

## 错误日志
```
粘贴相关错误日志
```

## 附加信息
任何其他相关信息
```

### 2. 功能请求

提出新功能时，包含：

- 功能描述
- 使用场景
- 期望行为
- 实现建议

#### 功能请求模板

```markdown
## 功能描述
简要描述新功能

## 问题解决
这个功能解决了什么问题

## 使用场景
描述具体使用场景

## 期望行为
详细描述功能行为

## 实现建议
如果有实现想法，请描述

## 附加信息
任何其他相关信息
```

## 🌍 社区参与

### 1. 讨论区

- 在GitHub Discussions中参与讨论
- 回答其他用户的问题
- 分享使用经验
- 提出改进建议

### 2. 代码审查

- 审查Pull Request
- 提供建设性反馈
- 帮助改进代码质量
- 学习最佳实践

### 3. 社区活动

- 组织线上/线下活动
- 参与会议和研讨会
- 撰写技术博客
- 制作教程视频

## 📋 贡献者指南

### 1. 成为维护者

长期贡献者可以成为项目维护者：

- 持续贡献高质量代码
- 积极参与社区讨论
- 帮助审查代码
- 改进项目文档

### 2. 贡献者认可

我们认可所有贡献者：

- 在README中列出贡献者
- 在发布说明中感谢贡献者
- 邀请活跃贡献者加入核心团队

## 🎯 贡献目标

### 短期目标

- 修复已知bug
- 改进文档
- 添加更多测试
- 优化性能

### 长期目标

- 添加新算法
- 支持更多数据格式
- 改进用户界面
- 扩展生态系统

## 📞 联系方式

- **GitHub Issues**: [项目Issues页面](https://github.com/your-username/Gold-Seeker/issues)
- **GitHub Discussions**: [项目讨论区](https://github.com/your-username/Gold-Seeker/discussions)
- **邮件**: dev@gold-seeker.com
- **QQ群**: 123456789
- **微信群**: 扫描二维码加入

---

感谢您对Gold-Seeker项目的贡献！每一个贡献都让这个项目变得更好。🎉

让我们一起构建更好的地球化学找矿预测平台！