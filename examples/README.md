# Gold-Seeker 使用示例

本目录包含Gold-Seeker平台的使用示例和演示代码。

## 示例文件

### 1. `complete_workflow.py`
**完整工作流演示** - 从原始地球化学数据到成矿预测的端到端流程

**功能演示：**
- 合成地球化学数据生成（模拟卡林型金矿特征）
- R型聚类分析识别元素共生组合
- 主成分分析推荐找矿元素
- 检测限数据处理和CLR变换
- C-A分形异常滤波
- 证据权计算与统计显著性检验
- 多证据层集成分析
- 响应面生成
- 结果可视化与报告输出

**运行方法：**
```bash
cd examples
python complete_workflow.py
```

**输出结果：**
- `../output/geochemical_data.csv` - 原始数据
- `../output/processed_data.csv` - 预处理后数据
- `../output/clr_data.csv` - CLR变换数据
- `../output/au_weights_of_evidence.csv` - Au证据权结果
- `../output/evidence_layers_summary.csv` - 证据层汇总
- `../output/r_mode_clustering.png` - R型聚类图
- `../output/pca_loadings.png` - PCA载荷图
- `../output/ca_fractal_analysis.png` - C-A分形分析图
- `../output/weights_of_evidence.png` - 证据权结果图

## 数据要求

### 输入数据格式
地球化学数据应为CSV格式，包含以下列：
- **元素列**: Au, As, Sb, Hg, Cu, Pb, Zn, Ag等
- **坐标列**: X, Y（可选，用于空间分析）
- **标签列**: Is_Deposit（1=矿点，0=非矿点，用于证据权计算）

### 数据质量要求
- 样品数量：建议 > 100个
- 元素覆盖：至少包含Au和2-3个指示元素
- 检测限信息：需要提供各元素的检测限值
- 训练点：至少10个已知矿点用于证据权训练

## 核心概念

### 1. 元素共生组合
基于Carranza (2009) 的R型聚类分析：
- **Au-As-Sb-Hg**: 典型的卡林型金矿组合
- **Cu-Pb-Zn**: 多金属硫化物组合
- **Ag-Au**: 低温热液组合

### 2. 分形异常滤波
使用C-A (Concentration-Area) 分形模型：
- **双对数图**: 浓度-面积关系
- **拐点检测**: 自动识别异常阈值
- **多重分形**: 背景与异常的分离

### 3. 证据权法
定量评价证据层的有效性：
- **W+**: 正权重（证据存在时的有利度）
- **W-**: 负权重（证据不存在时的不利度）
- **C**: 对比度（W+ - W-）
- **Studentized C**: 统计显著性检验

## 扩展示例

### 自定义数据示例
```python
import pandas as pd
from agents.tools.geochem import GeochemSelector

# 加载自己的数据
data = pd.read_csv('your_geochemical_data.csv')

# 设置检测限
detection_limits = {
    'Au': 0.05,
    'As': 0.5,
    'Sb': 0.2
}

# 创建选择器
selector = GeochemSelector(detection_limits)

# 执行R-mode分析
elements = ['Au', 'As', 'Sb', 'Hg', 'Cu', 'Pb', 'Zn']
result = selector.perform_r_mode_analysis(data, elements=elements)

# 查看结果
print(f"识别出 {len(result['clusters'])} 个元素组合")
```

### 集成到现有工作流
```python
from agents.spatial_analyst import SpatialAnalystAgent
from langchain_openai import ChatOpenAI

# 初始化智能体
llm = ChatOpenAI(model="gpt-4", temperature=0)
analyst = SpatialAnalystAgent(llm, detection_limits)

# 执行分析
result = analyst.analyze_geochemical_data(
    data=your_data,
    elements=['Au', 'As', 'Sb', 'Hg'],
    training_points=your_training_points
)

# 生成报告
report = analyst.generate_analysis_report(result, output_format="markdown")
print(report)
```

## 常见问题

### Q1: 如何处理大量低于检测限的数据？
A1: 系统提供多种方法：
- **substitution**: 检测限/2或检测限/√2替代
- **ROS**: Regression on Order Statistics
- **MLE**: 最大似然估计

### Q2: C-A分形分析没有明显拐点怎么办？
A2: 可以尝试：
- 调整分箱数量
- 使用不同的拐点检测方法
- 回退到统计方法（95百分位数）

### Q3: 证据权结果不显著怎么处理？
A3: 建议：
- 检查训练点质量
- 调整证据层分类
- 尝试其他元素组合
- 考虑空间相关性

## 技术支持

如需技术支持或报告问题，请：
1. 检查数据格式是否符合要求
2. 查看错误日志和警告信息
3. 参考Carranza (2009) 理论基础
4. 联系开发团队

## 参考文献

- Carranza, E.J.M. (2009). *Geochemical Anomaly and Mineral Prospectivity Mapping in GIS*. Handbook of Exploration and Environmental Geochemistry, Vol. 11. Elsevier.
- Cheng, Q., Agterberg, F.P., Ballantyne, S.B. (1994). The separation of geochemical anomalies from background by fractal methods. *Journal of Geochemical Exploration*, 51(2), 109-130.
- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall.