# Gold-Seeker 配置参考

本指南详细介绍Gold-Seeker平台的配置选项和参数设置。

## 📋 配置概览

Gold-Seeker使用YAML格式的配置文件，支持多层级配置结构：

```yaml
# 主配置文件 config.yaml
project:
  name: "金矿找矿预测项目"
  description: "基于Carranza理论的地球化学找矿预测"
  version: "1.0.0"

data:
  # 数据相关配置
  coordinate_system: "EPSG:4326"
  detection_limits: {}
  quality_checks: {}

analysis:
  # 分析参数配置
  element_selection: {}
  data_processing: {}
  anomaly_detection: {}
  weights_of_evidence: {}

modeling:
  # 机器学习配置
  algorithms: []
  hyperparameters: {}
  validation: {}

visualization:
  # 可视化配置
  plots: {}
  maps: {}
  export: {}

performance:
  # 性能配置
  memory: {}
  parallel: {}
  gpu: {}
```

## 📊 数据配置

### 基础数据配置

```yaml
data:
  # 坐标系统
  coordinate_system: "EPSG:4326"  # WGS84
  # 或
  coordinate_system: "EPSG:3857"  # Web Mercator
  
  # 数据格式
  format: "csv"  # csv, excel, geopackage, shapefile
  
  # 编码格式
  encoding: "utf-8"
  
  # 分隔符（CSV）
  delimiter: ","
  
  # 缺失值标记
  missing_values: ["", "NA", "null", "-999"]
```

### 检测限配置

```yaml
data:
  detection_limits:
    # 方法：fixed, adaptive, statistical
    method: "fixed"
    
    # 固定检测限
    fixed_limits:
      Au: 0.1
      Ag: 0.5
      Cu: 1.0
      Pb: 2.0
      Zn: 5.0
      As: 0.5
      Sb: 0.2
    
    # 自适应检测限
    adaptive_limits:
      method: "percentile"  # percentile, std_dev
      percentile: 5  # 5%分位数作为检测限
      std_multiplier: 2  # 2倍标准差
    
    # 统计检测限
    statistical_limits:
      method: "ros"  # ROS, Kaplan-Meier
      distribution: "lognormal"
```

### 数据质量检查

```yaml
data:
  quality_checks:
    # 完整性检查
    completeness:
      min_completeness: 0.8  # 最小完整性80%
      critical_columns: ["x", "y", "Au"]  # 关键列
    
    # 一致性检查
    consistency:
      coordinate_range:
        x: [0, 1000000]
        y: [0, 1000000]
      value_ranges:
        Au: [0, 100]
        Ag: [0, 1000]
        Cu: [0, 10000]
    
    # 异常值检查
    outlier_detection:
      method: "iqr"  # iqr, zscore, isolation_forest
      threshold: 3.0
      action: "flag"  # flag, remove, transform
```

## 🔬 分析配置

### 元素选择配置

```yaml
analysis:
  element_selection:
    # R型聚类分析
    r_mode_clustering:
      method: "ward"  # ward, complete, average, single
      distance_metric: "correlation"  # correlation, euclidean
      n_clusters: 5
    
    # 主成分分析
    pca:
      n_components: 0.95  # 解释95%方差
      rotation: "varimax"  # varimax, quartimax, equamax
    
    # 元素重要性
    importance:
      method: "combined"  # correlation, pca, mutual_info
      weights:
        correlation: 0.4
        pca: 0.3
        mutual_info: 0.3
    
    # 选择阈值
    selection_threshold: 0.7  # 重要性阈值
    max_elements: 10  # 最大选择元素数
```

### 数据处理配置

```yaml
analysis:
  data_processing:
    # 检测限处理
    censoring:
      method: "substitution"  # substitution, ros, kaplan_meier
      substitution_value: "half_detection_limit"
      
    # 数据变换
    transformation:
      method: "clr"  # clr, alr, ilr, log, sqrt
      centering: true
      scaling: true
      
    # 标准化
    standardization:
      method: "zscore"  # zscore, minmax, robust
      robust_quantile: 0.25
      
    # 异常值处理
    outlier_handling:
      method: "iqr"  # iqr, zscore, isolation_forest
      threshold: 3.0
      action: "transform"  # remove, transform, cap
```

### 异常检测配置

```yaml
analysis:
  anomaly_detection:
    # C-A分形分析
    fractal_analysis:
      method: "knee"  # knee, kmeans, piecewise_linear
      min_segments: 3
      max_segments: 10
      
    # 阈值确定
    threshold_methods:
      - "fractal"
      - "percentile"
      - "std_dev"
      
    # 百分位数阈值
    percentile_threshold:
      percentile: 95  # 95%分位数
      
    # 标准差阈值
    std_dev_threshold:
      multiplier: 2.0  # 2倍标准差
      
    # 异常分类
    anomaly_classification:
      method: "intensity"  # intensity, spatial, combined
      levels: ["low", "medium", "high", "extreme"]
```

### 证据权分析配置

```yaml
analysis:
  weights_of_evidence:
    # 权重计算
    weight_calculation:
      method: "binary"  # binary, continuous, fuzzy
      binary_threshold: "anomaly_threshold"
      
    # 连续权重
    continuous_weights:
      method: "logistic"  # logistic, spline, polynomial
      n_classes: 5
      
    # 模糊权重
    fuzzy_weights:
      membership_function: "sigmoid"  # sigmoid, gaussian, triangular
      parameters:
        a: 0.1
        b: 1.0
        c: 10.0
        
    # 显著性检验
    significance_test:
      method: "studentized"  # studentized, bootstrap
      alpha: 0.05  # 显著性水平
      n_bootstrap: 1000
```

## 🤖 机器学习配置

### 算法配置

```yaml
modeling:
  algorithms:
    # 随机森林
    random_forest:
      n_estimators: 100
      max_depth: null
      min_samples_split: 2
      min_samples_leaf: 1
      max_features: "sqrt"
      bootstrap: true
      random_state: 42
      
    # XGBoost
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
      
    # LightGBM
    lightgbm:
      n_estimators: 100
      max_depth: -1
      learning_rate: 0.1
      num_leaves: 31
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
      
    # 神经网络
    neural_network:
      hidden_layer_sizes: [100, 50]
      activation: "relu"
      solver: "adam"
      learning_rate: "constant"
      learning_rate_init: 0.001
      max_iter: 1000
      random_state: 42
```

### 超参数优化

```yaml
modeling:
  hyperparameter_optimization:
    # 优化方法
    method: "bayesian"  # grid, random, bayesian, genetic
    
    # 搜索空间
    search_space:
      random_forest:
        n_estimators: [50, 100, 200, 500]
        max_depth: [null, 10, 20, 30]
        min_samples_split: [2, 5, 10]
        
      xgboost:
        n_estimators: [50, 100, 200, 500]
        max_depth: [3, 6, 9, 12]
        learning_rate: [0.01, 0.1, 0.2]
        
    # 优化参数
    optimization:
      n_calls: 100
      n_initial_points: 10
      acq_func: "EI"  # EI, PI, LCB
      random_state: 42
```

### 模型验证

```yaml
modeling:
  validation:
    # 交叉验证
    cross_validation:
      method: "kfold"  # kfold, stratified, time_series
      n_splits: 5
      shuffle: true
      random_state: 42
      
    # 评估指标
    metrics:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1"
      - "roc_auc"
      - "confusion_matrix"
      
    # 验证策略
    validation_strategy:
      train_test_split: 0.2  # 20%测试集
      temporal_split: false  # 时间序列分割
      spatial_split: false  # 空间分割
      
    # 模型选择
    model_selection:
      criterion: "f1"  # 主要评估指标
      cv_scoring: "mean"  # mean, median, max
```

## 📊 可视化配置

### 图表配置

```yaml
visualization:
  plots:
    # 样式设置
    style: "seaborn"  # seaborn, matplotlib, plotly
    
    # 颜色配置
    colors:
      primary: "#1f77b4"
      secondary: "#ff7f0e"
      accent: "#2ca02c"
      background: "white"
      
    # 图表尺寸
    figure_size: [10, 6]
    dpi: 300
    
    # 字体设置
    font:
      family: "Arial"
      size: 12
      weight: "normal"
      
    # 图表类型
    chart_types:
      histogram:
        bins: 30
        density: true
        alpha: 0.7
        
      scatter:
        alpha: 0.6
        size: 50
        
      boxplot:
        showfliers: true
        notch: false
        
      heatmap:
        cmap: "viridis"
        center: 0
```

### 地图配置

```yaml
visualization:
  maps:
    # 底图
    basemap:
      provider: "openstreetmap"  # openstreetmap, cartodb, stamen
      style: "streets"  # streets, satellite, terrain
      
    # 地图样式
    style:
      color_scheme: "viridis"
      opacity: 0.7
      stroke_width: 1
      
    # 交互功能
    interactive:
      zoom: true
      pan: true
      tooltip: true
      legend: true
      
    # 导出设置
    export:
      format: "html"  # html, png, svg, pdf
      width: 1200
      height: 800
```

## ⚡ 性能配置

### 内存配置

```yaml
performance:
  memory:
    # 内存限制
    max_memory_usage: "8GB"
    
    # 分块处理
    chunk_size: 10000
    
    # 内存映射
    use_memory_mapping: true
    
    # 垃圾回收
    garbage_collection:
      frequency: "auto"  # auto, high, medium, low
      threshold: 0.8  # 80%内存使用时触发
```

### 并行配置

```yaml
performance:
  parallel:
    # 并行后端
    backend: "multiprocessing"  # multiprocessing, threading, joblib
    
    # 核心数
    n_jobs: -1  # -1表示使用所有核心
    
    # 并行策略
    strategy: "processes"  # processes, threads
    
    # 批处理
    batch_size: 1000
    
    # 负载均衡
    load_balancing: true
```

### GPU配置

```yaml
performance:
  gpu:
    # GPU使用
    use_gpu: true
    
    # GPU设备
    device: "cuda:0"  # cuda:0, cuda:1, cpu
    
    # 内存管理
    memory_fraction: 0.8  # 使用80% GPU内存
    
    # 混合精度
    mixed_precision: true
    
    # GPU加速算法
    accelerated_algorithms:
      - "xgboost"
      - "lightgbm"
      - "neural_network"
```

## 🔧 高级配置

### 日志配置

```yaml
logging:
  # 日志级别
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
  # 日志格式
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # 日志文件
  file: "gold_seeker.log"
  max_size: "10MB"
  backup_count: 5
  
  # 控制台输出
  console: true
  
  # 详细日志
  verbose: false
```

### 缓存配置

```yaml
cache:
  # 缓存启用
  enabled: true
  
  # 缓存目录
  directory: ".cache"
  
  # 缓存大小
  max_size: "1GB"
  
  # 缓存策略
  policy: "lru"  # lru, fifo, lfu
  
  # 缓存过期
  expiration: "7d"  # 7天
```

### 插件配置

```yaml
plugins:
  # 插件目录
  directory: "plugins"
  
  # 自动加载
  auto_load: true
  
  # 插件列表
  enabled:
    - "geochemical_plugin"
    - "visualization_plugin"
    - "ml_plugin"
    
  # 插件配置
  geochemical_plugin:
    version: "1.0.0"
    config_file: "plugins/geochemical.yaml"
```

## 🌐 环境配置

### 开发环境

```yaml
environment:
  # 环境类型
  type: "development"  # development, production, testing
  
  # 调试模式
  debug: true
  
  # 性能分析
  profiling: false
  
  # 实验性功能
  experimental_features: true
```

### 生产环境

```yaml
environment:
  # 环境类型
  type: "production"
  
  # 调试模式
  debug: false
  
  # 性能监控
  monitoring: true
  
  # 错误报告
  error_reporting: true
  
  # 安全设置
  security:
    encrypt_data: true
    secure_communication: true
```

## 📝 配置模板

### 基础模板

```yaml
# basic_config.yaml
project:
  name: "基础分析项目"
  
data:
  coordinate_system: "EPSG:4326"
  format: "csv"
  
analysis:
  element_selection:
    selection_threshold: 0.7
  data_processing:
    transformation:
      method: "clr"
  anomaly_detection:
    fractal_analysis:
      method: "knee"
      
modeling:
  algorithms:
    random_forest:
      n_estimators: 100
      
visualization:
  plots:
    style: "seaborn"
  maps:
    basemap:
      provider: "openstreetmap"
```

### 高级模板

```yaml
# advanced_config.yaml
project:
  name: "高级分析项目"
  description: "包含机器学习和高级可视化"
  
data:
  coordinate_system: "EPSG:4326"
  detection_limits:
    method: "adaptive"
    adaptive_limits:
      method: "percentile"
      percentile: 5
  quality_checks:
    completeness:
      min_completeness: 0.8
    
analysis:
  element_selection:
    r_mode_clustering:
      method: "ward"
      n_clusters: 5
    pca:
      n_components: 0.95
  data_processing:
    censoring:
      method: "ros"
    transformation:
      method: "clr"
  anomaly_detection:
    fractal_analysis:
      method: "piecewise_linear"
      min_segments: 3
  weights_of_evidence:
    weight_calculation:
      method: "continuous"
    significance_test:
      method: "studentized"
      alpha: 0.05
      
modeling:
  algorithms:
    random_forest:
      n_estimators: 200
      max_depth: 10
    xgboost:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
  hyperparameter_optimization:
    method: "bayesian"
    n_calls: 100
  validation:
    cross_validation:
      method: "stratified"
      n_splits: 10
      
visualization:
  plots:
    style: "seaborn"
    colors:
      primary: "#1f77b4"
      secondary: "#ff7f0e"
  maps:
    basemap:
      provider: "cartodb"
      style: "satellite"
    interactive:
      zoom: true
      pan: true
      tooltip: true
      
performance:
  memory:
    max_memory_usage: "16GB"
    chunk_size: 5000
  parallel:
    n_jobs: -1
    backend: "multiprocessing"
  gpu:
    use_gpu: true
    device: "cuda:0"
    
logging:
  level: "INFO"
  file: "gold_seeker.log"
  console: true
```

## 🔧 配置管理

### 加载配置

```python
from gold_seeker import GoldSeeker

# 从文件加载配置
gs = GoldSeeker(config_file="config.yaml")

# 从字典加载配置
config_dict = {
    "project": {"name": "测试项目"},
    "data": {"coordinate_system": "EPSG:4326"}
}
gs = GoldSeeker(config=config_dict)

# 合并多个配置
gs = GoldSeeker(
    config_file="base_config.yaml",
    overrides={"data": {"format": "excel"}}
)
```

### 动态配置

```python
# 运行时修改配置
gs.set_config("analysis.element_selection.selection_threshold", 0.8)

# 获取配置值
threshold = gs.get_config("analysis.element_selection.selection_threshold")

# 保存配置
gs.save_config("updated_config.yaml")
```

### 环境变量

```bash
# 设置环境变量
export GOLD_SEEKER_CONFIG_PATH="/path/to/config.yaml"
export GOLD_SEEKER_LOG_LEVEL="DEBUG"
export GOLD_SEEKER_MEMORY_LIMIT="8GB"
```

```python
# 使用环境变量
import os
from gold_seeker import GoldSeeker

config_path = os.getenv("GOLD_SEEKER_CONFIG_PATH", "default_config.yaml")
gs = GoldSeeker(config_file=config_path)
```

## 🎯 最佳实践

1. **分层配置**：使用基础配置+项目特定配置
2. **环境隔离**：开发、测试、生产环境使用不同配置
3. **版本控制**：配置文件纳入版本控制
4. **文档化**：为自定义配置添加注释
5. **验证**：使用配置验证工具确保配置正确

## 📚 相关文档

- [快速开始](quickstart.md)
- [基础教程](tutorial.md)
- [高级用法](advanced.md)
- [API参考](../development/api.md)

---

通过合理配置，您可以充分发挥Gold-Seeker的强大功能，满足各种地球化学找矿预测需求。