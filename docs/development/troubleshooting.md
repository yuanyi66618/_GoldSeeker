# Gold-Seeker 故障排除指南

本指南提供Gold-Seeker地球化学找矿预测智能平台的常见问题解决方案和故障排除方法。

## 📋 目录

- [快速诊断](#快速诊断)
- [安装问题](#安装问题)
- [配置问题](#配置问题)
- [数据问题](#数据问题)
- [分析问题](#分析问题)
- [性能问题](#性能问题)
- [网络问题](#网络问题)
- [系统问题](#系统问题)
- [调试工具](#调试工具)

## 🔍 快速诊断

### 1. 系统健康检查

```bash
# 运行系统诊断
gold-seeker doctor

# 检查配置
gold-seeker validate --config config/production.yaml

# 测试数据库连接
gold-seeker test --database

# 测试所有组件
gold-seeker test --all
```

### 2. 日志分析

```bash
# 查看实时日志
tail -f /var/log/gold-seeker/gold_seeker.log

# 查看错误日志
grep -i error /var/log/gold-seeker/gold_seeker.log

# 查看最近的警告
grep -i warning /var/log/gold-seeker/gold_seeker.log | tail -20

# 分析日志模式
awk '{print $1}' /var/log/gold-seeker/gold_seeker.log | sort | uniq -c | sort -nr
```

### 3. 性能监控

```bash
# 检查系统资源
top -p $(pgrep -f gold-seeker)
htop -p $(pgrep -f gold-seeker)

# 检查内存使用
ps aux | grep gold-seeker | awk '{sum+=$6} END {print "Memory:", sum/1024, "MB"}'

# 检查磁盘使用
df -h /var/lib/gold-seeker
du -sh /var/lib/gold-seeker/*
```

## 🛠️ 安装问题

### 问题1: 依赖安装失败

#### 症状
```
ERROR: Could not install packages due to an EnvironmentError
```

#### 解决方案

```bash
# 1. 升级pip
python -m pip install --upgrade pip

# 2. 清理缓存
pip cache purge

# 3. 使用国内镜像
pip install -e ".[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. 分步安装核心依赖
pip install numpy pandas scipy
pip install geopandas rasterio
pip install scikit-learn matplotlib
pip install -e ".[dev]"
```

#### GDAL安装问题

```bash
# Ubuntu/Debian
sudo apt-get install -y gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL==$(gdal-config --version)

# CentOS/RHEL
sudo yum install -y gdal gdal-devel
export GDAL_CONFIG=/usr/bin/gdal-config
pip install GDAL

# macOS
brew install gdal
pip install GDAL
```

### 问题2: 权限错误

#### 症状
```
PermissionError: [Errno 13] Permission denied
```

#### 解决方案

```bash
# 1. 使用用户安装
pip install --user -e ".[all]"

# 2. 修复权限
sudo chown -R $USER:$USER ~/.local
sudo chown -R $USER:$USER /usr/local/lib/python3.10/site-packages

# 3. 使用虚拟环境
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate
pip install -e ".[all]"
```

### 问题3: 版本冲突

#### 症状
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

#### 解决方案

```bash
# 1. 创建干净环境
python -m venv fresh-env
source fresh-env/bin/activate

# 2. 使用pip-tools
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt

# 3. 强制重新安装
pip install --force-reinstall --no-cache-dir -e ".[all]"
```

## ⚙️ 配置问题

### 问题1: 配置文件错误

#### 症状
```
ConfigError: Invalid configuration file
```

#### 诊断工具

```python
# config_validator.py
import yaml
from pathlib import Path

def validate_config(config_path):
    """验证配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查必需字段
        required_sections = ['project', 'data', 'analysis', 'logging']
        for section in required_sections:
            if section not in config:
                print(f"❌ 缺少必需部分: {section}")
                return False
        
        # 检查路径
        data_dir = Path(config['data']['data_dir'])
        if not data_dir.exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            return False
        
        print("✅ 配置文件验证通过")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ YAML语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False

if __name__ == "__main__":
    validate_config("config/default_config.yaml")
```

#### 常见配置错误

```yaml
# ❌ 错误: 缺少必需字段
project:
  name: "测试"  # 缺少environment字段

# ✅ 正确
project:
  name: "测试"
  environment: "development"

# ❌ 错误: 路径不存在
data:
  data_dir: "/nonexistent/path"

# ✅ 正确
data:
  data_dir: "./data"
```

### 问题2: 环境变量未设置

#### 症状
```
KeyError: 'SECRET_KEY'
```

#### 解决方案

```bash
# 1. 创建.env文件
cat > .env << EOF
SECRET_KEY=your-secret-key-here
DB_PASSWORD=your-database-password
API_KEY=your-api-key
EOF

# 2. 加载环境变量
export $(cat .env | xargs)

# 3. 在Python中加载
from dotenv import load_dotenv
load_dotenv()
```

### 问题3: 数据库连接失败

#### 症状
```
ConnectionError: Could not connect to database
```

#### 诊断脚本

```python
# db_test.py
import psycopg2
import redis

def test_postgres(config):
    """测试PostgreSQL连接"""
    try:
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            database=config['database']['name'],
            user=config['database']['user'],
            password=config['database']['password']
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"✅ PostgreSQL连接成功: {version}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL连接失败: {e}")
        return False

def test_redis(config):
    """测试Redis连接"""
    try:
        r = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            db=config['redis']['db']
        )
        
        r.ping()
        print("✅ Redis连接成功")
        return True
        
    except Exception as e:
        print(f"❌ Redis连接失败: {e}")
        return False
```

## 📊 数据问题

### 问题1: 数据格式错误

#### 症状
```
ValueError: Could not convert string to float
```

#### 数据诊断工具

```python
# data_diagnostic.py
import pandas as pd
import numpy as np
from pathlib import Path

def diagnose_data(file_path):
    """诊断数据文件"""
    try:
        # 尝试读取数据
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"❌ 不支持的文件格式: {file_path.suffix}")
            return False
        
        print(f"📊 数据形状: {df.shape}")
        print(f"📋 列名: {list(df.columns)}")
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("⚠️ 缺失值统计:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # 检查数据类型
        print("\n📝 数据类型:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # 检查数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n🔢 数值列统计:")
            print(df[numeric_cols].describe())
        
        # 检查异常值
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                print(f"⚠️ {col} 发现 {len(outliers)} 个异常值")
        
        print("✅ 数据诊断完成")
        return True
        
    except Exception as e:
        print(f"❌ 数据诊断失败: {e}")
        return False
```

#### 数据修复工具

```python
# data_fixer.py
import pandas as pd
import numpy as np

def fix_data_issues(df):
    """修复常见数据问题"""
    # 1. 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 2. 处理异常值
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 用边界值替换异常值
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # 3. 标准化列名
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    return df
```

### 问题2: 坐标系统错误

#### 症状
```
CRSError: Invalid coordinate reference system
```

#### 解决方案

```python
# crs_fixer.py
import geopandas as gpd
from pyproj import CRS

def fix_crs(gdf, target_crs='EPSG:4326'):
    """修复坐标系统"""
    try:
        # 检查当前CRS
        if gdf.crs is None:
            print("⚠️ 数据没有CRS信息，尝试自动检测...")
            # 假设数据是WGS84
            gdf.crs = 'EPSG:4326'
        
        # 转换到目标CRS
        if gdf.crs != target_crs:
            print(f"🔄 转换CRS: {gdf.crs} -> {target_crs}")
            gdf = gdf.to_crs(target_crs)
        
        # 验证几何有效性
        invalid_geoms = gdf[~gdf.geometry.is_valid]
        if len(invalid_geoms) > 0:
            print(f"⚠️ 发现 {len(invalid_geoms)} 个无效几何，尝试修复...")
            gdf.geometry = gdf.geometry.buffer(0)
        
        print("✅ CRS修复完成")
        return gdf
        
    except Exception as e:
        print(f"❌ CRS修复失败: {e}")
        return None
```

### 问题3: 数据量过大

#### 症状
```
MemoryError: Unable to allocate array
```

#### 解决方案

```python
# data_chunker.py
import pandas as pd
import numpy as np

def process_large_data(file_path, chunk_size=10000):
    """分块处理大数据"""
    chunks = []
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        print(f"处理块 {i+1}: {len(chunk)} 行")
        
        # 处理数据
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)
    
    # 合并结果
    result = pd.concat(chunks, ignore_index=True)
    return result

def process_chunk(chunk):
    """处理单个数据块"""
    # 实现具体的数据处理逻辑
    return chunk
```

## 🔬 分析问题

### 问题1: 模型训练失败

#### 症状
```
ValueError: Input contains NaN, infinity or a value too large
```

#### 诊断工具

```python
# model_diagnostic.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def diagnose_training_data(X, y):
    """诊断训练数据"""
    print("🔍 训练数据诊断")
    
    # 检查形状
    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")
    
    # 检查缺失值
    if np.isnan(X).any():
        print("❌ X包含NaN值")
        return False
    
    if np.isnan(y).any():
        print("❌ y包含NaN值")
        return False
    
    # 检查无穷值
    if np.isinf(X).any():
        print("❌ X包含无穷值")
        return False
    
    if np.isinf(y).any():
        print("❌ y包含无穷值")
        return False
    
    # 检查数据类型
    if not np.issubdtype(X.dtype, np.number):
        print("❌ X不是数值类型")
        return False
    
    if not np.issubdtype(y.dtype, np.number):
        print("❌ y不是数值类型")
        return False
    
    # 检查数据范围
    print(f"X范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"y范围: [{y.min():.3f}, {y.max():.3f}]")
    
    # 检查数据分布
    if len(np.unique(y)) < 2:
        print("❌ y只有一个类别")
        return False
    
    print("✅ 训练数据正常")
    return True

def preprocess_data(X, y):
    """预处理数据"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 处理异常值
    X_scaled = np.clip(X_scaled, -3, 3)
    
    return X_scaled, y, scaler
```

### 问题2: 分数计算错误

#### 症状
```
ZeroDivisionError: Division by zero in weights calculation
```

#### 解决方案

```python
# weights_fixer.py
import numpy as np
import pandas as pd

def safe_weights_calculation(evidence, target):
    """安全的权重计算"""
    try:
        # 计算基本统计量
        total_area = len(evidence)
        target_area = np.sum(target)
        non_target_area = total_area - target_area
        
        # 避免除零错误
        if target_area == 0:
            print("⚠️ 目标区域为空，无法计算权重")
            return None, None, None
        
        if non_target_area == 0:
            print("⚠️ 非目标区域为空，无法计算权重")
            return None, None, None
        
        # 计算权重
        w_plus = np.log((target_area / total_area) / (np.sum(evidence[target == 1]) / np.sum(evidence)))
        w_minus = np.log((non_target_area / total_area) / (np.sum(evidence[target == 0]) / np.sum(evidence)))
        contrast = w_plus - w_minus
        
        return w_plus, w_minus, contrast
        
    except Exception as e:
        print(f"❌ 权重计算失败: {e}")
        return None, None, None
```

### 问题3: 分形分析失败

#### 症状
```
LinAlgError: SVD did not converge
```

#### 解决方案

```python
# fractal_fixer.py
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

def robust_fractal_analysis(x, y):
    """鲁棒的分形分析"""
    try:
        # 移除无效值
        valid_idx = ~np.isnan(x) & ~np.isnan(y) & (x > 0) & (y > 0)
        x_clean = x[valid_idx]
        y_clean = y[valid_idx]
        
        if len(x_clean) < 3:
            print("❌ 有效数据点太少")
            return None, None, None
        
        # 对数变换
        log_x = np.log10(x_clean)
        log_y = np.log10(y_clean)
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        # 计算拟合优度
        y_pred = slope * log_x + intercept
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return slope, intercept, r_squared
        
    except Exception as e:
        print(f"❌ 分形分析失败: {e}")
        return None, None, None
```

## ⚡ 性能问题

### 问题1: 内存使用过高

#### 症状
```
MemoryError: Unable to allocate array
```

#### 解决方案

```python
# memory_optimizer.py
import gc
import psutil
import numpy as np
import pandas as pd

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    print(f"🧠 内存使用: {memory_mb:.1f} MB")
    return memory_mb

def optimize_memory(df):
    """优化DataFrame内存使用"""
    start_memory = monitor_memory()
    
    # 数值列优化
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 字符串列优化
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # 低基数
            df[col] = df[col].astype('category')
    
    # 强制垃圾回收
    gc.collect()
    
    end_memory = monitor_memory()
    print(f"💾 内存节省: {start_memory - end_memory:.1f} MB")
    
    return df

def process_with_memory_limit(func, data, max_memory_mb=1000):
    """在内存限制下处理数据"""
    def process_chunk(chunk):
        return func(chunk)
    
    # 如果数据太大，分块处理
    if isinstance(data, pd.DataFrame) and len(data) > 100000:
        chunk_size = 10000
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            result = process_chunk(chunk)
            results.append(result)
            
            # 检查内存使用
            if monitor_memory() > max_memory_mb:
                gc.collect()
        
        return pd.concat(results, ignore_index=True)
    else:
        return process_chunk(data)
```

### 问题2: 处理速度慢

#### 症状
```
处理时间过长，用户等待超时
```

#### 解决方案

```python
# performance_optimizer.py
import multiprocessing as mp
import numpy as np
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def parallel_apply(df, func, n_workers=None):
    """并行应用函数"""
    if n_workers is None:
        n_workers = mp.cpu_count() - 1
    
    # 分割数据
    chunks = np.array_split(df, n_workers)
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, chunks))
    
    # 合并结果
    return pd.concat(results, ignore_index=True)

def vectorized_operation(df, columns):
    """向量化操作"""
    # 使用numpy向量化操作替代循环
    data = df[columns].values
    
    # 示例：标准化
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized = (data - mean) / std
    
    result_df = df.copy()
    result_df[columns] = normalized
    
    return result_df

def cache_result(func):
    """缓存装饰器"""
    cache = {}
    
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        
        if key in cache:
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        
        return result
    
    return wrapper
```

### 问题3: 并发问题

#### 症状
```
DeadlockError: deadlock detected
```

#### 解决方案

```python
# concurrency_fixer.py
import threading
import queue
import time
from contextlib import contextmanager

class ThreadSafeCounter:
    """线程安全计数器"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self):
        with self._lock:
            return self._value

@contextmanager
def database_lock(db_connection, timeout=30):
    """数据库锁上下文管理器"""
    try:
        # 获取锁
        db_connection.execute("SELECT pg_advisory_lock(12345)")
        yield db_connection
    finally:
        # 释放锁
        db_connection.execute("SELECT pg_advisory_unlock(12345)")

def worker_with_retry(queue, result_queue, max_retries=3):
    """带重试的工作线程"""
    while True:
        try:
            task = queue.get(timeout=1)
            
            for attempt in range(max_retries):
                try:
                    result = process_task(task)
                    result_queue.put(result)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        result_queue.put(('error', str(e)))
                        break
                    time.sleep(2 ** attempt)  # 指数退避
            
            queue.task_done()
            
        except queue.Empty:
            break
```

## 🌐 网络问题

### 问题1: API连接失败

#### 症状
```
ConnectionError: Failed to establish connection
```

#### 解决方案

```python
# network_fixer.py
import requests
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def create_robust_session(max_retries=3, backoff_factor=0.3):
    """创建鲁棒的HTTP会话"""
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置超时
    session.timeout = (10, 30)  # 连接超时，读取超时
    
    return session

def safe_api_call(session, url, method='GET', **kwargs):
    """安全的API调用"""
    try:
        response = session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return None
```

### 问题2: 代理配置问题

#### 症状
```
ProxyError: HTTPSConnectionPool failed
```

#### 解决方案

```python
# proxy_fixer.py
import os
import requests

def configure_proxy():
    """配置代理"""
    # 从环境变量读取代理设置
    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')
    
    if http_proxy or https_proxy:
        proxies = {
            'http': http_proxy,
            'https': https_proxy
        }
        return proxies
    else:
        return None

def test_proxy_connection():
    """测试代理连接"""
    proxies = configure_proxy()
    
    try:
        response = requests.get(
            'https://httpbin.org/ip',
            proxies=proxies,
            timeout=10
        )
        print(f"✅ 代理连接成功: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ 代理连接失败: {e}")
        return False
```

## 🔧 系统问题

### 问题1: 服务启动失败

#### 症状
```
SystemExit: Error starting server
```

#### 诊断脚本

```bash
#!/bin/bash
# service_diagnostic.sh

echo "🔍 服务诊断开始..."

# 检查端口占用
echo "📡 检查端口占用:"
netstat -tlnp | grep :8000
netstat -tlnp | grep :8080

# 检查进程状态
echo "🔄 检查进程状态:"
ps aux | grep gold-seeker

# 检查系统资源
echo "💾 检查系统资源:"
free -h
df -h

# 检查日志
echo "📋 检查最近的错误日志:"
tail -20 /var/log/gold-seeker/gold_seeker.log | grep -i error

# 检查配置文件
echo "⚙️ 检查配置文件:"
gold-seeker validate --config /etc/gold-seeker/config.yaml

echo "🔍 服务诊断完成"
```

### 问题2: 权限问题

#### 症状
```
PermissionError: [Errno 13] Permission denied
```

#### 解决方案

```bash
#!/bin/bash
# permission_fix.sh

# 设置正确的文件权限
sudo chown -R gold-seeker:gold-seeker /var/lib/gold-seeker
sudo chmod -R 755 /var/lib/gold-seeker

# 设置日志目录权限
sudo chown -R gold-seeker:gold-seeker /var/log/gold-seeker
sudo chmod -R 755 /var/log/gold-seeker

# 设置配置文件权限
sudo chown gold-seeker:gold-seeker /etc/gold-seeker/config.yaml
sudo chmod 644 /etc/gold-seeker/config.yaml

# 添加用户到必要组
sudo usermod -aG docker gold-seeker
sudo usermod -aG redis gold-seeker
```

## 🛠️ 调试工具

### 1. 性能分析器

```python
# profiler.py
import cProfile
import pstats
import io
from functools import wraps

def profile_function(func):
    """函数性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建性能分析器
        pr = cProfile.Profile()
        
        # 开始分析
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        # 输出结果
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # 显示前10个最耗时的函数
        
        print(f"🔍 {func.__name__} 性能分析:")
        print(s.getvalue())
        
        return result
    
    return wrapper

# 使用示例
@profile_function
def slow_function():
    import time
    time.sleep(1)
    return "done"
```

### 2. 内存分析器

```python
# memory_profiler.py
import tracemalloc
from functools import wraps

def memory_profile(func):
    """内存使用分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 开始内存跟踪
        tracemalloc.start()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 获取内存使用情况
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"🧠 {func.__name__} 内存使用:")
        print(f"  当前: {current / 1024 / 1024:.1f} MB")
        print(f"  峰值: {peak / 1024 / 1024:.1f} MB")
        
        return result
    
    return wrapper
```

### 3. 日志增强器

```python
# logger_enhancer.py
import logging
import traceback
import functools
import time

def enhanced_logger(logger_name='gold_seeker'):
    """增强的日志记录器"""
    logger = logging.getLogger(logger_name)
    
    # 创建详细格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    # 添加处理器
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def log_exceptions(func):
    """异常日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = enhanced_logger()
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.info(f"✅ {func.__name__} 成功完成 ({end_time - start_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ {func.__name__} 失败: {str(e)}")
            logger.error(f"📋 异常堆栈:\n{traceback.format_exc()}")
            raise
    
    return wrapper
```

### 4. 系统监控器

```python
# system_monitor.py
import psutil
import time
import threading
from datetime import datetime

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, interval=60):
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        """开始监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            self._log_system_status()
            time.sleep(self.interval)
    
    def _log_system_status(self):
        """记录系统状态"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        
        # 网络IO
        network = psutil.net_io_counters()
        
        print(f"📊 {timestamp} 系统状态:")
        print(f"  CPU: {cpu_percent}%")
        print(f"  内存: {memory.percent}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
        print(f"  磁盘: {disk.percent}% ({disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB)")
        print(f"  网络: ↑{network.bytes_sent/1024/1024:.1f}MB ↓{network.bytes_recv/1024/1024:.1f}MB")

# 使用示例
monitor = SystemMonitor(interval=30)
monitor.start()

# 运行你的代码...

monitor.stop()
```

---

通过使用这些故障排除工具和解决方案，您可以快速诊断和解决Gold-Seeker平台运行中遇到的各种问题。记住，良好的日志记录和监控是预防问题的关键。