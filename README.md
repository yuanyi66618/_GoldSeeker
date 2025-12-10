# Gold-Seeker: 金矿智能预测智能体平台

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-Latest-brightgreen.svg)](docs/)

融合地质领域知识与先进大模型技术的金矿智能预测智能体平台，集成多智能体架构与LangChain技术，实现从原始地球化学数据到成矿预测的端到端自动化分析。平台结合深度学习、知识图谱和专家系统，为金矿勘探提供智能化解决方案。

## 🌟 核心特性

### 🤖 多智能体架构
- **CoordinatorAgent**: 任务协调与工作流管理
- **ArchivistAgent**: 知识管理与GraphRAG集成
- **SpatialAnalystAgent**: 地球化学空间分析（核心）
- **ModelerAgent**: 机器学习建模与预测
- **CriticAgent**: 结果验证与报告生成

### 🔬 地球化学分析工具
- **GeochemSelector**: R型聚类分析与元素重要性排序
- **GeochemProcessor**: 检测限数据处理与CLR变换
- **FractalAnomalyFilter**: C-A分形异常滤波
- **WeightsOfEvidenceCalculator**: 证据权计算与统计检验

### 🧠 AI增强分析
- LangChain集成与Chain-of-Thought推理
- 自动化工作流编排
- 智能参数优化
- 专家知识库集成

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 8GB+ RAM（推荐16GB）
- 支持CUDA的GPU（可选，用于加速）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker
```

2. **创建虚拟环境**
```bash
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate  # Linux/Mac
gold-seeker-env\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，添加你的API密钥
```

### 快速体验

```python
from agents.spatial_analyst import SpatialAnalystAgent
from langchain_openai import ChatOpenAI
import pandas as pd

# 初始化智能体
llm = ChatOpenAI(model="gpt-4", temperature=0)
analyst = SpatialAnalystAgent(llm)

# 加载数据
data = pd.read_csv('data/geochemical_samples.csv')

# 执行分析
result = analyst.analyze_geochemical_data(
    data=data,
    elements=['Au', 'As', 'Sb', 'Hg'],
    training_points=training_data
)

# 生成报告
report = analyst.generate_analysis_report(result)
print(report)
```

## 📊 使用示例

### 完整工作流演示
```bash
cd examples
python complete_workflow.py
```

该示例展示了从合成数据生成到成矿预测的完整流程，包括：
- 卡林型金矿地球化学特征模拟
- R型聚类分析识别元素共生组合
- C-A分形异常滤波
- 证据权计算与统计检验
- 多证据层集成与响应面生成

### 自定义分析
```python
from agents.tools.geochem import GeochemSelector, GeochemProcessor

# 元素选择分析
selector = GeochemSelector(detection_limits)
clusters = selector.perform_r_mode_analysis(data, elements=['Au', 'As', 'Sb', 'Hg'])

# 数据预处理
processor = GeochemProcessor(detection_limits)
processed_data = processor.transform_clr(data, elements=['Au', 'As', 'Sb'])
```

## 🏗️ 项目结构

```
Gold-Seeker/
├── agents/                    # 智能体模块
│   ├── coordinator.py        # 任务协调智能体
│   ├── archivist.py          # 知识管理智能体
│   ├── spatial_analyst.py    # 空间分析智能体
│   ├── modeler.py            # 建模智能体
│   └── critic.py             # 验证智能体
├── agents/tools/geochem/      # 地球化学工具
│   ├── selector.py           # 元素选择工具
│   ├── processor.py          # 数据处理工具
│   ├── fractal.py            # 分形分析工具
│   └── woe.py                # 证据权计算工具
├── tests/                     # 单元测试
├── examples/                  # 使用示例
├── docs/                      # 文档
├── config/                    # 配置文件
└── data/                      # 示例数据
```

## 📚 核心理论

### Carranza (2009) 方法论

本平台严格遵循Carranza (2009) 提出的地球化学异常与成矿预测理论框架：

1. **元素共生组合分析**
   - R型聚类分析识别元素关联性
   - 主成分分析确定找矿指示元素
   - 地质背景下的元素组合解释

2. **分形异常滤波**
   - C-A (Concentration-Area) 分形模型
   - 背景与异常的定量分离
   - 多重分形分析技术

3. **证据权建模**
   - 二元证据层生成
   - W+、W-、对比度C计算
   - Studentized C统计显著性检验

4. **空间集成分析**
   - 多证据层权重优化
   - 响应面生成与可视化
   - 不确定性评估

## 🛠️ 开发指南

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_geochem_tools.py

# 生成覆盖率报告
pytest --cov=agents tests/
```

### 代码格式化
```bash
# 格式化代码
black agents/ tests/ examples/

# 检查代码风格
flake8 agents/ tests/ examples/

# 类型检查
mypy agents/
```

### 文档生成
```bash
cd docs
make html
```

## 📈 性能优化

### 大数据集处理
- 使用Dask进行分布式计算
- GPU加速（CuPy支持）
- 内存优化策略

### 并行计算
```python
from joblib import Parallel, delayed

# 并行处理多个元素
results = Parallel(n_jobs=-1)(
    delayed(process_element)(data, element) 
    for element in elements
)
```

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 开发规范
- 遵循PEP 8代码风格
- 添加适当的单元测试
- 更新相关文档
- 确保所有测试通过

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Carranza, E.J.M.** (2009) - 理论基础与方法论指导
- **Cheng, Q.** 等 - 分形地球化学理论贡献
- **LangChain团队** - AI框架支持
- **GeoPandas/PySAL社区** - 地理空间分析工具

## 📞 联系我们

- **项目主页**: https://github.com/your-username/Gold-Seeker
- **文档**: https://gold-seeker.readthedocs.io/
- **问题反馈**: https://github.com/your-username/Gold-Seeker/issues
- **邮箱**: your-email@example.com

## 🗺️ 路线图

### v1.0 (当前版本)
- ✅ 核心地球化学分析工具
- ✅ 多智能体架构
- ✅ LangChain集成
- ✅ 完整工作流示例

### v1.1 (计划中)
- 🔄 GraphRAG知识库集成
- 🔄 QGIS/ArcGIS插件
- 🔄 Web界面开发
- 🔄 实时数据流处理

### v2.0 (未来版本)
- 📋 深度学习模型集成
- 📋 多源数据融合
- 📋 云端部署支持
- 📋 移动端应用

---

**Gold-Seeker** - 让地球化学找矿预测更智能、更精准、更高效！

- ✅ GeochemSelector (自动特征筛选)
- ✅ GeochemProcessor (数据清洗与分形)
- ✅ FractalAnomalyFilter (C-A分形模型)
- ✅ WeightsOfEvidenceCalculator (空间评价)
- ✅ SpatialAnalystAgent (LangChain集成)