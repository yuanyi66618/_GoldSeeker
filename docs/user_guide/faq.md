# Gold-Seeker 常见问题

本文档收集了Gold-Seeker用户常见的问题和解决方案。

## 📋 目录

- [安装问题](#安装问题)
- [数据问题](#数据问题)
- [分析问题](#分析问题)
- [可视化问题](#可视化问题)
- [性能问题](#性能问题)
- [错误排查](#错误排查)
- [高级问题](#高级问题)

## 🚀 安装问题

### Q: 安装时出现"Microsoft Visual C++ 14.0 is required"错误

**A:** 这是Windows系统上常见的编译错误，解决方案：

```bash
# 方法1：安装预编译包
pip install --only-binary=all gold-seeker

# 方法2：安装Visual C++ Build Tools
# 下载地址：https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 方法3：使用conda安装
conda install -c conda-forge gold-seeker
```

### Q: GDAL安装失败

**A:** GDAL是地理空间数据处理的核心依赖，安装方法：

```bash
# Windows
conda install -c conda-forge gdal

# macOS
brew install gdal

# Linux (Ubuntu/Debian)
sudo apt-get install gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

### Q: 内存不足安装失败

**A:** 使用用户安装或虚拟环境：

```bash
# 用户安装
pip install --user gold-seeker

# 虚拟环境
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate  # Linux/Mac
gold-seeker-env\Scripts\activate  # Windows
pip install gold-seeker
```

### Q: 权限错误

**A:** 使用管理员权限或用户安装：

```bash
# 管理员权限（Windows）
右键点击命令提示符 → "以管理员身份运行"

# 用户安装
pip install --user gold-seeker

# 或使用sudo（Linux/Mac）
sudo pip install gold-seeker
```

## 📊 数据问题

### Q: 支持哪些数据格式？

**A:** Gold-Seeker支持多种格式：

```python
# CSV文件
data = gs.load_data("data.csv")

# Excel文件
data = gs.load_data("data.xlsx", sheet_name="Sheet1")

# GeoPackage
data = gs.load_data("data.gpkg")

# Shapefile
data = gs.load_data("data.shp")

# GeoJSON
data = gs.load_data("data.geojson")

# 直接从DataFrame加载
import pandas as pd
df = pd.read_csv("data.csv")
data = gs.load_data(df)
```

### Q: 数据格式要求是什么？

**A:** 基本要求：

```csv
x,y,Au,Ag,Cu,Pb,Zn,As,Sb,Censoring
1000,2000,0.5,2.1,15.3,8.7,1.2,12.4,2.1,0
1100,2100,1.2,3.5,18.9,12.4,2.1,15.6,3.2,0
```

- **坐标字段**：`x`, `y` 或 `X`, `Y`
- **元素含量**：至少一个目标元素
- **检测限标记**：`Censoring`（可选，0=检测到，1=低于检测限）

### Q: 如何处理缺失值？

**A:** 多种处理方法：

```python
# 自动处理
data = gs.handle_missing_values(
    data=data,
    method="auto"  # auto, drop, fill, interpolate
)

# 指定方法
data = gs.handle_missing_values(
    data=data,
    method="interpolation",
    columns=["Au", "Ag", "Cu"]
)

# 自定义填充值
data = gs.handle_missing_values(
    data=data,
    method="fill",
    fill_value={"Au": 0.1, "Ag": 0.5}
)
```

### Q: 如何处理异常值？

**A:** 异常值检测和处理：

```python
# 检测异常值
outliers = gs.detect_outliers(
    data=data,
    method="iqr",  # iqr, zscore, isolation_forest
    threshold=3.0
)

# 处理异常值
data = gs.handle_outliers(
    data=data,
    outliers=outliers,
    method="transform"  # remove, transform, cap
)
```

### Q: 数据量太大怎么办？

**A:** 大数据处理策略：

```python
# 分块处理
gs = GoldSeeker(chunk_size=10000)
results = gs.analyze_large_dataset(data)

# 内存映射
gs = GoldSeeker(use_memory_mapping=True)

# 并行处理
gs = GoldSeeker(n_jobs=4)
results = gs.parallel_analyze(data)
```

## 🔬 分析问题

### Q: 元素选择结果不合理？

**A:** 调整元素选择参数：

```python
# 调整选择阈值
config = {
    "analysis": {
        "element_selection": {
            "selection_threshold": 0.8,  # 提高阈值
            "max_elements": 5  # 限制元素数量
        }
    }
}

gs = GoldSeeker(config=config)
```

### Q: C-A分形分析失败？

**A:** 检查数据质量和参数：

```python
# 数据质量检查
quality = gs.assess_data_quality(data)
print(quality)

# 调整分形参数
config = {
    "analysis": {
        "anomaly_detection": {
            "fractal_analysis": {
                "method": "kmeans",  # 尝试不同方法
                "min_segments": 2,
                "max_segments": 8
            }
        }
    }
}
```

### Q: 证据权分析结果为NaN？

**A:** 常见原因和解决方案：

```python
# 检查数据完整性
print(data.isnull().sum())

# 检查异常阈值
anomalies = gs.detect_anomalies(data, "Au")
print(f"异常样本数: {anomalies.sum()}")

# 调整阈值方法
config = {
    "analysis": {
        "weights_of_evidence": {
            "weight_calculation": {
                "method": "continuous"  # 使用连续权重
            }
        }
    }
}
```

### Q: 机器学习模型性能差？

**A:** 模型优化策略：

```python
# 数据预处理
data = gs.preprocess_for_ml(
    data=data,
    target_element="Au",
    feature_selection=True,
    scaling=True
)

# 超参数优化
config = {
    "modeling": {
        "hyperparameter_optimization": {
            "method": "bayesian",
            "n_calls": 100
        }
    }
}

# 交叉验证
results = gs.cross_validate(
    data=data,
    target_element="Au",
    cv_folds=10
)
```

## 📈 可视化问题

### Q: 地图不显示？

**A:** 检查坐标系统和数据：

```python
# 检查坐标系统
print(data.crs)

# 转换坐标系统
data = data.to_crs("EPSG:4326")

# 检查数据范围
print(data.total_bounds)

# 创建简单地图测试
gs.plot_simple_map(data, "Au")
```

### Q: 图表显示异常？

**A:** 检查数据和配置：

```python
# 检查数据类型
print(data.dtypes)

# 转换数据类型
data["Au"] = pd.to_numeric(data["Au"], errors="coerce")

# 调整图表配置
config = {
    "visualization": {
        "plots": {
            "style": "matplotlib",  # 尝试不同样式
            "figure_size": [12, 8]
        }
    }
}
```

### Q: 交互式地图无法加载？

**A:** 检查依赖和网络：

```python
# 检查依赖
import plotly
import folium
print(f"Plotly版本: {plotly.__version__}")
print(f"Folium版本: {folium.__version__}")

# 使用静态地图
gs.plot_static_map(data, "Au", output_file="static_map.png")
```

## ⚡ 性能问题

### Q: 分析速度太慢？

**A:** 性能优化策略：

```python
# 并行处理
gs = GoldSeeker(
    n_jobs=-1,  # 使用所有CPU核心
    backend="multiprocessing"
)

# GPU加速
gs = GoldSeeker(
    use_gpu=True,
    device="cuda:0"
)

# 内存优化
gs = GoldSeeker(
    chunk_size=5000,
    max_memory_usage="4GB"
)
```

### Q: 内存不足错误？

**A:** 内存管理：

```python
# 分块处理
gs = GoldSeeker(chunk_size=1000)

# 内存映射
gs = GoldSeeker(use_memory_mapping=True)

# 垃圾回收
import gc
gc.collect()

# 监控内存使用
memory_usage = gs.monitor_memory()
print(f"内存使用: {memory_usage.current}MB")
```

### Q: 磁盘空间不足？

**A:** 磁盘空间管理：

```python
# 清理缓存
gs.clear_cache()

# 压缩结果
gs.compress_results(output_file="compressed_results.zip")

# 删除中间文件
gs.cleanup_temp_files()
```

## 🔍 错误排查

### Q: 如何获取详细错误信息？

**A:** 启用详细日志：

```python
# 启用调试模式
gs = GoldSeeker(debug=True)

# 设置日志级别
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看错误详情
try:
    results = gs.analyze(data)
except Exception as e:
    print(f"错误类型: {type(e)}")
    print(f"错误信息: {str(e)}")
    import traceback
    traceback.print_exc()
```

### Q: ImportError: No module named 'gold_seeker'

**A:** 安装和路径问题：

```bash
# 检查安装
pip list | grep gold-seeker

# 重新安装
pip uninstall gold-seeker
pip install gold-seeker

# 检查Python路径
python -c "import sys; print(sys.path)"
```

### Q: KeyError: 'column_name'

**A:** 列名问题：

```python
# 检查列名
print(data.columns)

# 重命名列
data = data.rename(columns={
    "longitude": "x",
    "latitude": "y",
    "gold": "Au"
})

# 检查大小写
data.columns = data.columns.str.upper()
```

### Q: ValueError: cannot convert float NaN to integer

**A:** NaN值处理：

```python
# 检查NaN值
print(data.isnull().sum())

# 删除NaN值
data = data.dropna()

# 填充NaN值
data = data.fillna(0)

# 转换数据类型
data = data.astype({"Au": "float32"})
```

## 🚀 高级问题

### Q: 如何自定义分析流程？

**A:** 自定义工作流：

```python
from gold_seeker import WorkflowPlan, Task

# 创建自定义工作流
workflow = WorkflowPlan("自定义分析")

# 添加任务
workflow.add_task(Task(
    name="数据预处理",
    tool="GeochemProcessor",
    parameters={"method": "custom"}
))

# 执行工作流
results = gs.execute_workflow(workflow, data)
```

### Q: 如何集成外部数据？

**A:** 多源数据融合：

```python
# 加载外部数据
geology = gs.load_data("geology.shp")
geophysics = gs.load_data("geophysics.tif")

# 数据融合
fused_data = gs.fuse_data(
    geochem_data=data,
    geology_data=geology,
    geophysics_data=geophysics
)
```

### Q: 如何部署到服务器？

**A:** 服务器部署：

```python
# 创建API服务
from gold_seeker import create_api

app = create_api(gs)

# 运行服务
app.run(host="0.0.0.0", port=8080)
```

### Q: 如何扩展功能？

**A:** 插件开发：

```python
from gold_seeker.plugins import BasePlugin

class CustomPlugin(BasePlugin):
    def initialize(self, gs_instance):
        # 插件初始化
        pass
    
    def process(self, data):
        # 自定义处理逻辑
        return data

# 注册插件
gs.register_plugin(CustomPlugin)
```

## 📞 获取帮助

### 在线资源

- 📖 [完整文档](../README.md)
- 🐛 [GitHub Issues](https://github.com/your-username/Gold-Seeker/issues)
- 💬 [社区讨论](https://github.com/your-username/Gold-Seeker/discussions)
- 📧 技术支持: support@gold-seeker.com

### 问题报告

提交问题时请包含：

1. **Gold-Seeker版本**
2. **Python版本**
3. **操作系统**
4. **错误信息**
5. **重现步骤**
6. **最小示例代码**

### 社区支持

- 加入用户群组
- 参与开源贡献
- 分享使用经验
- 提供改进建议

---

**如果您的问题未在此文档中找到答案，请随时联系我们的技术支持团队！** 🎉

我们致力于为用户提供最好的地球化学找矿预测解决方案。