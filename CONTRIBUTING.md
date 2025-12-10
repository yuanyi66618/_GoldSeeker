# 贡献指南

感谢您对Gold-Seeker项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- ✨ 开发新功能
- 🧪 编写测试
- 🌍 翻译和本地化

## 目录

1. [开发环境设置](#开发环境设置)
2. [贡献流程](#贡献流程)
3. [代码规范](#代码规范)
4. [测试指南](#测试指南)
5. [文档贡献](#文档贡献)
6. [问题报告](#问题报告)
7. [功能请求](#功能请求)
8. [社区准则](#社区准则)

## 开发环境设置

### 前置要求

- Python 3.9+
- Git
- 推荐使用虚拟环境

### 快速开始

1. **Fork项目**
   ```bash
   # 在GitHub上Fork项目，然后克隆
   git clone https://github.com/your-username/Gold-Seeker.git
   cd Gold-Seeker
   ```

2. **设置开发环境**
   ```bash
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   
   # 安装开发依赖
   make install-dev
   # 或
   pip install -e ".[dev]"
   ```

3. **配置环境变量**
   ```bash
   # 复制环境变量模板
   cp .env.example .env
   
   # 编辑.env文件，添加必要的API密钥
   # 注意：不要提交真实的API密钥到版本控制
   ```

4. **验证安装**
   ```bash
   # 运行测试
   make test
   
   # 运行示例
   make run-example
   ```

### 开发工具

我们推荐使用以下工具：

- **IDE**: VS Code, PyCharm, 或其他支持Python的IDE
- **代码格式化**: Black, isort
- **代码检查**: flake8, mypy
- **测试**: pytest
- **文档**: Sphinx

## 贡献流程

### 1. 创建分支

```bash
# 从main分支创建新分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

### 2. 开发和测试

```bash
# 进行开发...
# 确保代码通过所有测试
make test

# 检查代码质量
make lint

# 格式化代码
make format
```

### 3. 提交更改

```bash
# 添加更改
git add .

# 提交（使用有意义的提交信息）
git commit -m "feat: 添加新的地球化学分析功能"

# 推送到Fork
git push origin feature/your-feature-name
```

### 4. 创建Pull Request

1. 在GitHub上创建Pull Request
2. 填写PR模板
3. 等待代码审查
4. 根据反馈进行修改

### 5. 合并

- 通过所有检查后，维护者将合并您的PR
- 感谢您的贡献！🎉

## 代码规范

### Python代码风格

我们遵循以下规范：

- **PEP 8**: Python代码风格指南
- **Black**: 代码格式化工具
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

#### 示例

```python
"""
模块文档字符串
简短描述模块功能。

详细描述...
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def analyze_geochemical_data(
    data: pd.DataFrame,
    elements: List[str],
    detection_limits: Dict[str, float],
    method: str = "default"
) -> Dict[str, any]:
    """
    分析地球化学数据。
    
    Args:
        data: 地球化学数据
        elements: 要分析的元素列表
        detection_limits: 检测限字典
        method: 分析方法
        
    Returns:
        分析结果字典
        
    Raises:
        ValueError: 当输入数据无效时
    """
    if data.empty:
        raise ValueError("数据不能为空")
    
    # 实现分析逻辑...
    return {"result": "success"}
```

### 提交信息规范

我们使用[约定式提交](https://www.conventionalcommits.org/zh-hans/)格式：

```
<类型>[可选的作用域]: <描述>

[可选的正文]

[可选的脚注]
```

#### 类型

- `feat`: 新功能
- `fix`: bug修复
- `docs`: 文档更新
- `style`: 代码格式化（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

#### 示例

```bash
feat(geochem): 添加C-A分形分析功能

实现了基于Cheng et al. (1994)的C-A分形异常检测方法，
支持多种拐点检测算法。

Closes #123
```

## 测试指南

### 测试结构

```
tests/
├── conftest.py          # pytest配置和fixtures
├── test_config.py       # 配置管理测试
├── test_utils.py        # 工具函数测试
├── test_agents.py       # 智能体测试
├── test_geochem_tools.py # 地球化学工具测试
├── test_integration.py  # 集成测试
└── data/               # 测试数据
    ├── sample_data.csv
    └── expected_results/
```

### 编写测试

```python
import pytest
import pandas as pd
from agents.tools.geochem import GeochemSelector


class TestGeochemSelector:
    """地球化学选择器测试类"""
    
    def test_r_mode_analysis(self, sample_geochemical_data):
        """测试R型聚类分析"""
        selector = GeochemSelector({'Au': 0.05, 'As': 0.5})
        
        result = selector.perform_r_mode_analysis(
            sample_geochemical_data, 
            ['Au', 'As', 'Sb', 'Hg']
        )
        
        assert 'clusters' in result
        assert len(result['clusters']) > 0
```

### 运行测试

```bash
# 运行所有测试
make test

# 运行特定测试
pytest tests/test_geochem_tools.py -v

# 运行带覆盖率的测试
make test-cov

# 运行集成测试
make test-integration
```

### 测试覆盖率

我们要求：

- 新功能必须有对应的测试
- 测试覆盖率不低于80%
- 关键功能覆盖率不低于95%

## 文档贡献

### 文档类型

- **API文档**: 代码中的docstring
- **用户指南**: 使用教程和示例
- **开发文档**: 架构设计和开发指南
- **部署文档**: 安装和配置说明

### 文档风格

- 使用清晰、简洁的语言
- 提供代码示例
- 包含图表和截图
- 保持文档与代码同步

### 构建文档

```bash
# 构建文档
make docs

# 本地预览
make docs-serve
```

## 问题报告

### 报告bug

使用GitHub Issues报告bug，请包含：

1. **问题描述**: 清晰描述遇到的问题
2. **复现步骤**: 详细的重现步骤
3. **期望行为**: 描述期望的正确行为
4. **实际行为**: 描述实际发生的情况
5. **环境信息**: 
   - 操作系统
   - Python版本
   - Gold-Seeker版本
   - 相关依赖版本
6. **错误信息**: 完整的错误堆栈
7. **最小示例**: 能复现问题的最小代码示例

### Bug报告模板

```markdown
## Bug描述
简短描述bug

## 复现步骤
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## 期望行为
清晰简洁地描述期望发生的情况

## 实际行为
清晰简洁地描述实际发生的情况

## 环境信息
- OS: [例如 Windows 10, macOS 11.0, Ubuntu 20.04]
- Python版本: [例如 3.9.0]
- Gold-Seeker版本: [例如 1.0.0]

## 错误信息
```
粘贴完整的错误堆栈
```

## 附加信息
添加任何其他有助于解决问题的信息
```

## 功能请求

### 提出新功能

1. **检查现有功能**: 确保功能不存在
2. **搜索已有请求**: 避免重复
3. **详细描述**: 清晰描述功能需求
4. **使用场景**: 说明使用场景和价值
5. **实现建议**: 如果有想法，提供实现建议

### 功能请求模板

```markdown
## 功能描述
清晰简洁地描述想要的功能

## 问题解决
这个功能解决了什么问题？

## 建议的解决方案
描述您希望如何实现这个功能

## 替代方案
描述您考虑过的其他替代解决方案

## 使用场景
描述这个功能的具体使用场景

## 附加信息
添加任何其他相关信息或截图
```

## 社区准则

### 行为准则

我们致力于为每个人提供友好、安全和欢迎的环境，无论：

- 经验水平
- 性别认同和表达
- 性取向
- 残疾
- 个人外貌
- 身体大小
- 种族
- 民族
- 年龄
- 宗教
- 国籍

### 期望行为

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 不当行为

- 使用性化的语言或图像
- 人身攻击或政治攻击
- 公开或私下骚扰
- 未经明确许可发布他人的私人信息
- 其他在专业环境中可能被认为不当的行为

## 获得帮助

如果您需要帮助或有疑问：

1. **查看文档**: [在线文档](https://gold-seeker.readthedocs.io/)
2. **搜索Issues**: 查看是否已有类似问题
3. **创建Discussion**: 在GitHub Discussions中提问
4. **联系维护者**: 通过邮件联系项目维护者

## 认可贡献者

我们感谢所有贡献者的努力！贡献者将被列在：

- README.md的贡献者部分
- 发布说明中
- 项目网站上

## 许可证

通过贡献代码，您同意您的贡献将在[MIT许可证](LICENSE)下授权。

---

再次感谢您的贡献！🎉