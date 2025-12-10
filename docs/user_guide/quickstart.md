# Gold-Seeker 快速开始

欢迎使用Gold-Seeker地球化学找矿预测智能平台！本指南将帮助您在5分钟内快速上手。

## 🚀 快速安装

### 方法1：使用pip安装（推荐）

```bash
# 创建虚拟环境
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate  # Linux/Mac
# 或
gold-seeker-env\Scripts\activate  # Windows

# 安装Gold-Seeker
pip install gold-seeker

# 验证安装
gold-seeker --version
```

### 方法2：从源码安装

```bash
# 克隆仓库
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker

# 安装依赖
pip install -e .

# 验证安装
python -m gold_seeker --version
```

## 📊 第一次分析

### 1. 准备数据

创建一个简单的CSV文件 `sample_data.csv`：

```csv
x,y,Au,Ag,Cu,As,Sb
1000,2000,0.5,2.1,15.3,8.7,1.2
1100,2100,1.2,3.5,18.9,12.4,2.1
1200,2200,0.8,2.8,16.7,9.8,1.5
1300,2300,2.1,4.2,22.1,15.6,3.2
1400,2400,0.3,1.9,14.2,7.1,0.9
```

### 2. 运行分析

```bash
# 基础地球化学分析
gold-seeker analyze --data sample_data.csv --elements Au Ag Cu --output results/

# 完整工作流
gold-seeker workflow --data sample_data.csv --config config/default_config.yaml --output workflow_results/
```

### 3. 查看结果

分析完成后，结果将保存在指定目录中：

```
results/
├── analysis_report.html      # 交互式报告
├── geochemical_anomalies.tif # 异常图
├── element_importance.png    # 元素重要性图
└── processing_summary.json   # 处理摘要
```

## 🎯 常用命令

### 数据分析

```bash
# 分析特定元素
gold-seeker analyze --data data.csv --elements Au Ag --method clr

# 使用自定义配置
gold-seeker analyze --data data.csv --config my_config.yaml

# 生成交互式报告
gold-seeker analyze --data data.csv --elements Au --report interactive
```

### 工作流管理

```bash
# 运行完整工作流
gold-seeker workflow --data data.csv --elements Au Ag Cu As Sb

# 验证数据质量
gold-seeker validate --data data.csv --quality-check

# 查看系统信息
gold-seeker info
```

### 示例和测试

```bash
# 运行示例
gold-seeker example --name carlin_type --output example_results/

# 运行测试
gold-seeker test --quick
```

## 📈 快速可视化

### Python脚本示例

```python
from gold_seeker import GeochemProcessor, FractalAnomalyFilter
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('sample_data.csv')

# 处理数据
processor = GeochemProcessor()
processed_data = processor.transform_clr(data[['Au', 'Ag', 'Cu']])

# 异常检测
filter_anomaly = FractalAnomalyFilter()
anomalies = filter_anomaly.filter_anomalies(processed_data, 'Au')

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(data['x'], data['y'], c=anomalies, cmap='Reds', s=50)
plt.colorbar(label='异常强度')
plt.title('金元素地球化学异常')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.show()
```

### Jupyter Notebook

```python
# 在Jupyter中运行
%load_ext gold_seeker.jupyter

# 快速分析
%gold_seeker analyze --data sample_data.csv --elements Au Ag
```

## ⚙️ 基本配置

### 创建配置文件

```bash
# 生成默认配置
gold-seeker init --config my_config.yaml

# 编辑配置
nano my_config.yaml
```

### 配置示例

```yaml
# 数据处理配置
data:
  coordinate_columns: ['x', 'y']
  detection_limits:
    Au: 0.1
    Ag: 0.5
    Cu: 1.0

# 分析参数
analysis:
  transformation: 'clr'
  outlier_method: 'iqr'
  fractal_method: 'knee'

# 输出设置
output:
  format: ['geojson', 'shapefile']
  visualization: true
  report: 'interactive'
```

## 🔧 常见问题解决

### 安装问题

**问题**: `ImportError: No module named 'gold_seeker'`

**解决方案**:
```bash
# 确保虚拟环境已激活
which python
pip install -e .

# 或重新安装
pip uninstall gold-seeker
pip install gold-seeker
```

**问题**: 依赖安装失败

**解决方案**:
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gold-seeker

# 或安装特定版本
pip install gold-seeker==1.0.0
```

### 数据问题

**问题**: 数据加载失败

**解决方案**:
```bash
# 检查数据格式
gold-seeker validate --data your_data.csv --format-check

# 查看数据信息
gold-seeker info --data your_data.csv
```

**问题**: 坐标系统错误

**解决方案**:
```python
# 在配置中指定坐标系统
config = {
    'data': {
        'coordinate_columns': ['x', 'y'],
        'crs': 'EPSG:4326'  # WGS84
    }
}
```

### 性能问题

**问题**: 处理大数据集时内存不足

**解决方案**:
```yaml
# 在配置中启用分块处理
processing:
  chunk_size: 10000
  memory_limit: '4GB'
  parallel: true
```

## 📚 下一步

恭喜！您已经完成了Gold-Seeker的快速开始。接下来您可以：

1. 📖 阅读[基础教程](tutorial.md)深入了解
2. 🔬 查看[示例集合](../examples/README.md)学习实际应用
3. 🎯 了解[高级用法](advanced.md)掌握专业技巧
4. 📚 学习[理论基础](../theory/carranza.md)理解方法原理

## 🆘 获取帮助

如果您在使用过程中遇到问题：

- 📖 查看[常见问题](faq.md)
- 🔍 搜索[GitHub Issues](https://github.com/your-username/Gold-Seeker/issues)
- 💬 参与[GitHub Discussions](https://github.com/your-username/Gold-Seeker/discussions)
- 📧 发送邮件到support@gold-seeker.com

---

**开始您的地球化学找矿预测之旅！** 🚀

Gold-Seeker让复杂的地球化学分析变得简单高效。祝您使用愉快！