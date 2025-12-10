# Gold-Seeker 安装指南

本指南将详细介绍如何在不同操作系统上安装和配置Gold-Seeker地球化学找矿预测智能平台。

## 📋 系统要求

### 最低要求

- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.9 或更高版本
- **内存**: 4GB RAM（推荐8GB+）
- **存储**: 2GB可用空间
- **网络**: 用于下载依赖包

### 推荐配置

- **操作系统**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.10 或 3.11
- **内存**: 16GB RAM
- **存储**: 10GB可用空间
- **GPU**: 支持CUDA的GPU（可选，用于加速计算）

## 🔧 Python环境准备

### 1. 安装Python

#### Windows

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载Python 3.9+版本
3. 运行安装程序，勾选"Add Python to PATH"
4. 验证安装：

```cmd
python --version
pip --version
```

#### macOS

```bash
# 使用Homebrew安装
brew install python@3.10

# 或从官网下载安装包
# https://www.python.org/downloads/macos/

# 验证安装
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian)

```bash
# 更新包列表
sudo apt update

# 安装Python和pip
sudo apt install python3 python3-pip python3-venv

# 验证安装
python3 --version
pip3 --version
```

### 2. 创建虚拟环境

强烈建议使用虚拟环境来隔离项目依赖：

```bash
# 创建虚拟环境
python -m venv gold-seeker-env

# 激活虚拟环境
# Windows
gold-seeker-env\Scripts\activate

# macOS/Linux
source gold-seeker-env/bin/activate
```

## 📦 安装方法

### 方法1：使用pip安装（推荐）

这是最简单和推荐的安装方法：

```bash
# 安装最新版本
pip install gold-seeker

# 安装特定版本
pip install gold-seeker==1.0.0

# 安装开发版本
pip install gold-seeker[dev]
```

#### 验证安装

```bash
# 检查版本
gold-seeker --version

# 查看帮助
gold-seeker --help

# 运行测试
gold-seeker test --quick
```

### 方法2：从源码安装

适用于开发者或需要最新功能的用户：

```bash
# 克隆仓库
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker

# 安装依赖
pip install -r requirements.txt

# 安装包（开发模式）
pip install -e .

# 或安装发布版本
pip install .
```

### 方法3：使用conda安装

如果您使用Anaconda或Miniconda：

```bash
# 创建conda环境
conda create -n gold-seeker python=3.10
conda activate gold-seeker

# 安装依赖
conda install -c conda-forge geopandas rasterio scikit-learn

# 安装Gold-Seeker
pip install gold-seeker
```

## 🔌 可选依赖

### 完整功能安装

```bash
# 安装所有可选依赖
pip install gold-seeker[complete]

# 或单独安装特定功能
pip install gold-seeker[ml]          # 机器学习功能
pip install gold-seeker[visualization] # 高级可视化
pip install gold-seeker[parallel]     # 并行计算
pip install gold-seeker[dev]          # 开发工具
```

### GPU支持

如果您有NVIDIA GPU并希望加速计算：

```bash
# 安装CUDA支持
pip install gold-seeker[gpu]

# 验证GPU支持
python -c "import gold_seeker; print(gold_seeker.gpu_available())"
```

### 地理信息系统支持

```bash
# 安装GIS相关依赖
pip install gold-seeker[gis]

# 这将安装：
# - GDAL
# - Fiona
# - Shapely
# - PyProj
```

## 🌐 网络配置

### 使用国内镜像

如果您在中国大陆，建议使用国内镜像加速下载：

```bash
# 清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gold-seeker

# 阿里云镜像
pip install -i https://mirrors.aliyun.com/pypi/simple gold-seeker

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 代理设置

如果您需要通过代理访问网络：

```bash
# 临时设置代理
pip install --proxy http://user:password@proxy.server:port gold-seeker

# 永久配置
pip config set global.proxy http://proxy.server:port
```

## 🐳 Docker安装

### 使用预构建镜像

```bash
# 拉取镜像
docker pull goldseeker/gold-seeker:latest

# 运行容器
docker run -it --rm -v $(pwd):/data goldseeker/gold-seeker:latest

# 在容器中运行分析
gold-seeker analyze --data /data/sample.csv --elements Au Ag
```

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker

# 构建镜像
docker build -t gold-seeker .

# 运行容器
docker run -it --rm -v $(pwd):/data gold-seeker
```

## 📱 特定平台安装

### Windows

#### 使用Chocolatey

```powershell
# 安装Python
choco install python

# 安装Gold-Seeker
pip install gold-seeker
```

#### 使用WSL

```bash
# 在WSL中安装
sudo apt update
sudo apt install python3 python3-pip
pip install gold-seeker
```

### macOS

#### 使用MacPorts

```bash
# 安装Python
sudo port install python310

# 安装Gold-Seeker
pip install gold-seeker
```

### Linux

#### Ubuntu/Debian

```bash
# 安装系统依赖
sudo apt install python3-dev python3-pip build-essential

# 安装Gold-Seeker
pip install gold-seeker
```

#### CentOS/RHEL/Fedora

```bash
# CentOS/RHEL
sudo yum install python3-devel python3-pip gcc

# Fedora
sudo dnf install python3-devel python3-pip gcc

# 安装Gold-Seeker
pip install gold-seeker
```

## 🔍 故障排除

### 常见安装问题

#### 问题1：Python版本不兼容

**错误信息**：
```
ERROR: Package 'gold-seeker' requires a different Python
```

**解决方案**：
```bash
# 检查Python版本
python --version

# 升级Python或使用兼容版本
# 重新创建虚拟环境
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate
pip install gold-seeker
```

#### 问题2：依赖安装失败

**错误信息**：
```
ERROR: Could not install packages due to an EnvironmentError
```

**解决方案**：
```bash
# 升级pip
pip install --upgrade pip

# 使用用户安装
pip install --user gold-seeker

# 或使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gold-seeker
```

#### 问题3：GDAL安装失败

**错误信息**：
```
ERROR: Could not find GDAL
```

**解决方案**：

**Windows**:
```cmd
# 使用conda安装
conda install -c conda-forge gdal

# 或下载预编译轮子
pip install GDAL-3.4.1-cp39-cp39-win_amd64.whl
```

**Linux**:
```bash
# 安装系统依赖
sudo apt install libgdal-dev gdal-bin

# 设置环境变量
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# 安装Python包
pip install GDAL
```

**macOS**:
```bash
# 使用Homebrew
brew install gdal

# 设置环境变量
export GDAL_LIBRARY_PATH=$(brew --prefix gdal)/lib/libgdal.dylib
export GEOS_LIBRARY_PATH=$(brew --prefix geos)/lib/libgeos_c.dylib

# 安装Python包
pip install GDAL
```

#### 问题4：权限错误

**错误信息**：
```
ERROR: Could not install packages due to PermissionError
```

**解决方案**：
```bash
# 使用用户安装
pip install --user gold-seeker

# 或使用虚拟环境
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate
pip install gold-seeker
```

### 验证安装

#### 基本验证

```bash
# 检查版本
gold-seeker --version

# 查看帮助
gold-seeker --help

# 运行简单测试
gold-seeker test --quick
```

#### 功能验证

```python
# 创建测试脚本
import gold_seeker
from gold_seeker import GeochemProcessor

# 测试基本功能
processor = GeochemProcessor()
print("Gold-Seeker安装成功！")
print(f"版本: {gold_seeker.__version__}")
```

#### 依赖验证

```bash
# 检查关键依赖
python -c "
import numpy, pandas, geopandas, sklearn, matplotlib
print('所有依赖安装成功！')
"
```

## 🔄 更新和维护

### 更新Gold-Seeker

```bash
# 更新到最新版本
pip install --upgrade gold-seeker

# 更新到特定版本
pip install gold-seeker==1.1.0

# 检查可用版本
pip index versions gold-seeker
```

### 卸载

```bash
# 卸载Gold-Seeker
pip uninstall gold-seeker

# 清理缓存
pip cache purge
```

## 📚 下一步

安装完成后，您可以：

1. 🚀 开始[快速开始](quickstart.md)
2. 📖 阅读[基础教程](tutorial.md)
3. 🔬 查看[示例集合](../examples/README.md)
4. ⚙️ 了解[配置选项](configuration.md)

## 🆘 获取帮助

如果您在安装过程中遇到问题：

- 📖 查看[常见问题](faq.md)
- 🔍 搜索[GitHub Issues](https://github.com/your-username/Gold-Seeker/issues)
- 💬 参与[GitHub Discussions](https://github.com/your-username/Gold-Seeker/discussions)
- 📧 发送邮件到install@gold-seeker.com

---

**安装成功！** 🎉

现在您可以开始使用Gold-Seeker进行地球化学找矿预测分析了。祝您使用愉快！