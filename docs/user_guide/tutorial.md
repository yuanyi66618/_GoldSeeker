# Gold-Seeker 基础教程

本教程将带您从零开始学习Gold-Seeker地球化学找矿预测智能平台的使用方法。

## 📚 教程大纲

1. [平台概述](#平台概述)
2. [数据准备](#数据准备)
3. [基础分析](#基础分析)
4. [高级功能](#高级功能)
5. [结果解读](#结果解读)
6. [实战案例](#实战案例)

## 🎯 平台概述

### 什么是Gold-Seeker？

Gold-Seeker是一个基于Carranza (2009)理论的地球化学找矿预测智能平台，提供：

- **智能元素选择**：基于R型聚类和主成分分析
- **数据处理**：检测限处理、对数比变换、异常值检测
- **异常识别**：C-A分形分析、多种阈值方法
- **证据权分析**：统计显著性检验、对比度计算
- **机器学习建模**：多种算法集成、交叉验证
- **可视化展示**：交互式地图、统计图表

### 核心工作流程

```
数据输入 → 元素选择 → 数据处理 → 异常识别 → 证据权分析 → 找矿预测
```

### 四大智能代理

1. **协调代理 (Coordinator)**：任务调度和工作流管理
2. **档案代理 (Archivist)**：知识管理和图谱检索
3. **空间分析代理 (Spatial Analyst)**：地球化学数据处理和分析
4. **建模代理 (Modeler)**：机器学习建模和预测
5. **评估代理 (Critic)**：结果验证和报告生成

## 📊 数据准备

### 数据格式要求

Gold-Seeker支持多种数据格式，推荐使用CSV格式：

```csv
x,y,Au,Ag,Cu,Pb,Zn,As,Sb,Censoring
1000,2000,0.5,2.1,15.3,8.7,1.2,12.4,2.1,0
1100,2100,1.2,3.5,18.9,12.4,2.1,15.6,3.2,0
1200,2200,0.8,2.8,16.7,9.8,1.5,9.8,1.5,0
1300,2300,2.1,4.2,22.1,15.6,3.2,18.9,2.8,0
1400,2400,0.3,1.9,14.2,7.1,0.9,7.1,0.8,0
```

### 必需字段

- **坐标字段**：`x`, `y` 或 `X`, `Y`
- **元素含量**：至少包含一个目标元素（如 `Au`）
- **检测限标记**：`Censoring`（可选，0=检测到，1=低于检测限）

### 数据质量检查

```python
import pandas as pd
from gold_seeker.utils import validate_data

# 加载数据
data = pd.read_csv("geochemical_data.csv")

# 数据质量检查
validation_result = validate_data(data)
print(validation_result)

# 查看基本统计信息
print(data.describe())

# 检查缺失值
print(data.isnull().sum())
```

## 🔧 基础分析

### 1. 初始化平台

```python
from gold_seeker import GoldSeeker

# 创建平台实例
gs = GoldSeeker()

# 加载配置（可选）
gs.load_config("config.yaml")
```

### 2. 数据加载

```python
# 从CSV文件加载
data = gs.load_data("geochemical_data.csv")

# 从DataFrame加载
import pandas as pd
df = pd.read_csv("data.csv")
data = gs.load_data(df)

# 查看数据信息
print(data.info())
print(data.head())
```

### 3. 快速分析

```python
# 运行快速分析
results = gs.quick_analyze(
    data=data,
    target_element="Au",
    area_name="研究区域"
)

# 查看分析摘要
print(results.summary())
```

### 4. 分步分析

```python
# 步骤1：元素选择
selector = gs.get_tool("GeochemSelector")
selected_elements = selector.select_elements(
    data=data,
    target_element="Au",
    method="r_mode_clustering"
)

print("选择的元素:", selected_elements)

# 步骤2：数据处理
processor = gs.get_tool("GeochemProcessor")
processed_data = processor.process_data(
    data=data,
    elements=selected_elements,
    censoring_method="substitution",
    transform_method="clr"
)

# 步骤3：异常识别
fractal_filter = gs.get_tool("FractalAnomalyFilter")
anomalies = fractal_filter.filter_anomalies(
    data=processed_data,
    target_element="Au",
    method="knee"
)

# 步骤4：证据权分析
woe_calculator = gs.get_tool("WeightsOfEvidenceCalculator")
woe_results = woe_calculator.calculate_weights(
    data=processed_data,
    target_element="Au",
    anomaly_threshold=anomalies.threshold
)
```

## 🚀 高级功能

### 1. 自定义配置

```python
# 创建自定义配置
config = {
    "data": {
        "coordinate_system": "EPSG:4326",
        "detection_limits": {
            "Au": 0.1,
            "Ag": 0.5,
            "Cu": 1.0
        }
    },
    "analysis": {
        "outlier_method": "iqr",
        "clr_transform": True,
        "fractal_method": "knee",
        "significance_level": 0.05
    },
    "modeling": {
        "ml_models": ["random_forest", "xgboost"],
        "cross_validation": 5,
        "feature_selection": True
    }
}

# 应用配置
gs = GoldSeeker(config=config)
```

### 2. 机器学习建模

```python
# 训练预测模型
modeler = gs.get_agent("Modeler")
model = modeler.train_model(
    data=processed_data,
    target_element="Au",
    features=selected_elements,
    model_type="random_forest"
)

# 进行预测
predictions = modeler.predict_probability(
    data=processed_data,
    model=model
)

# 模型验证
validation = modeler.validate_model(
    model=model,
    test_data=processed_data
)
```

### 3. 批量处理

```python
# 处理多个元素
elements = ["Au", "Ag", "Cu", "Pb", "Zn"]
results = {}

for element in elements:
    result = gs.quick_analyze(
        data=data,
        target_element=element,
        area_name=f"研究区域-{element}"
    )
    results[element] = result

# 比较结果
gs.compare_results(results)
```

## 📈 结果解读

### 1. 分析报告

```python
# 生成详细报告
report = gs.generate_report(
    results=results,
    format="html",
    output_file="analysis_report.html"
)

# 查看关键指标
print("关键发现:")
for key, value in results.key_findings.items():
    print(f"  {key}: {value}")
```

### 2. 可视化结果

```python
# 生成图表
gs.plot_results(
    results=results,
    plot_types=["histogram", "scatter", "correlation"],
    output_dir="plots/"
)

# 创建交互式地图
map_html = gs.create_interactive_map(
    data=data,
    results=results,
    output_file="interactive_map.html"
)
```

### 3. 结果导出

```python
# 导出为GeoJSON
gs.export_results(
    results=results,
    format="geojson",
    output_file="results.geojson"
)

# 导出为Shapefile
gs.export_results(
    results=results,
    format="shapefile",
    output_file="results.shp"
)
```

## 🎯 实战案例

### 案例1：卡林型金矿预测

```python
# 加载卡林型金矿数据
data = gs.load_data("carlin_type_gold_data.csv")

# 配置卡林型金矿特定参数
config = {
    "analysis": {
        "target_elements": ["Au", "As", "Sb", "Hg", "Tl"],
        "pathfinder_elements": ["As", "Sb", "Hg", "Tl"],
        "fractal_method": "piecewise_linear"
    }
}

# 运行分析
gs = GoldSeeker(config=config)
results = gs.quick_analyze(
    data=data,
    target_element="Au",
    area_name="卡林型金矿区"
)

# 生成找矿预测图
gs.create_prospectivity_map(
    results=results,
    output_file="carlin_prospectivity_map.html"
)
```

### 案例2：斑岩型铜矿预测

```python
# 加载斑岩型铜矿数据
data = gs.load_data("porphyry_copper_data.csv")

# 配置斑岩型铜矿特定参数
config = {
    "analysis": {
        "target_elements": ["Cu", "Mo", "Au", "Ag"],
        "pathfinder_elements": ["Mo", "Re", "Bi"],
        "fractal_method": "kmeans"
    }
}

# 运行分析
gs = GoldSeeker(config=config)
results = gs.quick_analyze(
    data=data,
    target_element="Cu",
    area_name="斑岩型铜矿区"
)

# 集成多种证据层
evidence_layers = gs.integrate_evidence(
    results=results,
    method="fuzzy_logic"
)
```

### 案例3：区域尺度评价

```python
# 处理大区域数据
data = gs.load_data("regional_geochemistry.csv")

# 分块处理大数据集
chunk_size = 10000
results = []

for i in range(0, len(data), chunk_size):
    chunk = data.iloc[i:i+chunk_size]
    result = gs.quick_analyze(
        data=chunk,
        target_element="Au",
        area_name=f"区域-{i//chunk_size}"
    )
    results.append(result)

# 合并结果
combined_results = gs.merge_results(results)

# 生成区域评价报告
gs.generate_regional_report(
    results=combined_results,
    output_file="regional_assessment.html"
)
```

## 🔧 常见问题解决

### 1. 数据问题

```python
# 处理缺失值
data = gs.handle_missing_values(
    data=data,
    method="interpolation"
)

# 处理异常值
data = gs.handle_outliers(
    data=data,
    method="iqr",
    threshold=3.0
)
```

### 2. 性能优化

```python
# 启用并行处理
gs = GoldSeeker(
    parallel=True,
    n_jobs=4
)

# 使用内存优化
gs = GoldSeeker(
    memory_limit="4GB",
    chunk_size=1000
)
```

### 3. 结果验证

```python
# 交叉验证
cv_results = gs.cross_validate(
    data=data,
    target_element="Au",
    cv_folds=5
)

# 专家验证
expert_review = gs.expert_review(
    results=results,
    expert_criteria="industry_standard"
)
```

## 📚 进阶学习

完成本教程后，您可以：

1. 阅读[高级用法指南](advanced.md)
2. 查看[配置参考](configuration.md)
3. 学习[理论基础](../theory/carranza.md)
4. 探索[更多示例](../examples/README.md)

## 🎉 总结

恭喜！您已经掌握了Gold-Seeker的基础使用方法。现在您可以：

- ✅ 准备和加载地球化学数据
- ✅ 执行基础的找矿预测分析
- ✅ 解读和可视化分析结果
- ✅ 处理实际案例数据
- ✅ 解决常见问题

继续探索Gold-Seeker的更多高级功能，发现找矿预测的无限可能！

## 🆘 获取帮助

- 📖 [用户指南](../user_guide/README.md)
- 🐛 [GitHub Issues](https://github.com/your-username/Gold-Seeker/issues)
- 💬 [社区讨论](https://github.com/your-username/Gold-Seeker/discussions)
- 📧 support@gold-seeker.com