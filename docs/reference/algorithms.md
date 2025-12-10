# 算法参考

本文档详细介绍Gold-Seeker平台中使用的各种算法实现，包括数学原理、计算步骤和代码实现。

## 📊 地球化学数据处理算法

### 1. 删失数据处理算法

#### 1.1 检测限/2方法 (DL/2)

**数学原理**:
对于低于检测限的值，使用检测限的一半进行替换：

$$x_{imputed} = \frac{DL}{2}$$

其中：
- $x_{imputed}$: 插补后的值
- $DL$: 检测限

**代码实现**:
```python
def dl_over_2_imputation(data, detection_limits):
    """
    检测限/2方法插补删失数据
    
    参数:
    - data: 地球化学数据 (DataFrame)
    - detection_limits: 检测限字典
    
    返回:
    - imputed_data: 插补后的数据
    """
    imputed_data = data.copy()
    
    for element, dl in detection_limits.items():
        if element in data.columns:
            censored_mask = data[element] < dl
            imputed_data.loc[censored_mask, element] = dl / 2
    
    return imputed_data
```

#### 1.2 Rosner方法

**数学原理**:
基于正态分布假设，使用最大似然估计计算删失数据的期望值：

$$E[X|X < DL] = \mu - \sigma \cdot \frac{\phi(z)}{\Phi(z)}$$

其中：
- $\mu$: 非删失数据的均值
- $\sigma$: 非删失数据的标准差
- $z = \frac{DL - \mu}{\sigma}$: 标准化检测限
- $\phi(z)$: 标准正态分布的概率密度函数
- $\Phi(z)$: 标准正态分布的累积分布函数

**代码实现**:
```python
import numpy as np
from scipy.stats import norm

def rosner_imputation(data, element, detection_limit):
    """
    Rosner方法插补删失数据
    
    参数:
    - data: 地球化学数据
    - element: 元素名称
    - detection_limit: 检测限
    
    返回:
    - imputed_value: 插补值
    """
    # 获取非删失数据
    uncensored_data = data[data[element] >= detection_limit][element]
    
    if len(uncensored_data) == 0:
        return detection_limit / 2
    
    # 估计参数
    mu = np.mean(uncensored_data)
    sigma = np.std(uncensored_data)
    
    # 计算标准化检测限
    z = (detection_limit - mu) / sigma
    
    # 计算期望值
    expected_value = mu - sigma * norm.pdf(z) / norm.cdf(z)
    
    return max(expected_value, detection_limit / 100)  # 避免负值或零值
```

#### 1.3 最大似然估计方法

**数学原理**:
对于删失数据，构建似然函数：

$$L(\mu, \sigma) = \prod_{i \in U} f(x_i; \mu, \sigma) \cdot \prod_{j \in C} F(DL; \mu, \sigma)$$

其中：
- $U$: 非删失数据集合
- $C$: 删失数据集合
- $f(x; \mu, \sigma)$: 正态分布概率密度函数
- $F(x; \mu, \sigma)$: 正态分布累积分布函数

**代码实现**:
```python
from scipy.optimize import minimize
from scipy.stats import norm

def maximum_likelihood_imputation(data, element, detection_limit):
    """
    最大似然估计方法插补删失数据
    
    参数:
    - data: 地球化学数据
    - element: 元素名称
    - detection_limit: 检测限
    
    返回:
    - imputed_value: 插补值
    """
    # 分离删失和非删失数据
    uncensored = data[data[element] >= detection_limit][element]
    censored_count = len(data[data[element] < detection_limit])
    
    def negative_log_likelihood(params):
        mu, sigma = params
        
        # 非删失数据的对数似然
        log_likelihood = np.sum(norm.logpdf(uncensored, mu, sigma))
        
        # 删失数据的对数似然
        log_likelihood += censored_count * norm.logcdf(detection_limit, mu, sigma)
        
        return -log_likelihood
    
    # 初始参数估计
    initial_params = [np.mean(uncensored), np.std(uncensored)]
    
    # 优化
    result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B')
    
    if result.success:
        mu_opt, sigma_opt = result.x
        # 计算删失数据的期望值
        z = (detection_limit - mu_opt) / sigma_opt
        expected_value = mu_opt - sigma_opt * norm.pdf(z) / norm.cdf(z)
        return max(expected_value, detection_limit / 100)
    else:
        # 如果优化失败，回退到Rosner方法
        return rosner_imputation(data, element, detection_limit)
```

### 2. 数据转换算法

#### 2.1 中心对数比转换 (CLR)

**数学原理**:
对于成分数据 $\mathbf{x} = (x_1, x_2, ..., x_D)$，CLR转换定义为：

$$clr(x_i) = \ln\left(\frac{x_i}{g(\mathbf{x})}\right)$$

其中几何均值：
$$g(\mathbf{x}) = \left(\prod_{j=1}^{D} x_j\right)^{1/D}$$

**代码实现**:
```python
import numpy as np

def clr_transform(data):
    """
    中心对数比转换
    
    参数:
    - data: 成分数据 (DataFrame或numpy数组)
    
    返回:
    - clr_data: CLR转换后的数据
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    # 添加小常数避免对数零
    epsilon = 1e-10
    data_array = data_array + epsilon
    
    # 计算几何均值
    geometric_mean = np.exp(np.mean(np.log(data_array), axis=1))
    
    # CLR转换
    clr_data = np.log(data_array / geometric_mean[:, np.newaxis])
    
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(clr_data, index=data.index, columns=data.columns)
    else:
        return clr_data

def clr_inverse_transform(clr_data):
    """
    CLR逆转换
    
    参数:
    - clr_data: CLR转换后的数据
    
    返回:
    - original_data: 原始成分数据
    """
    if isinstance(clr_data, pd.DataFrame):
        clr_array = clr_data.values
    else:
        clr_array = clr_data
    
    # 逆转换
    exp_clr = np.exp(clr_array)
    original_data = exp_clr / np.sum(exp_clr, axis=1, keepdims=True)
    
    if isinstance(clr_data, pd.DataFrame):
        return pd.DataFrame(original_data, index=clr_data.index, columns=clr_data.columns)
    else:
        return original_data
```

#### 2.2 加法对数比转换 (ALR)

**数学原理**:
选择参考成分 $x_D$，ALR转换定义为：

$$alr(x_i) = \ln\left(\frac{x_i}{x_D}\right), \quad i = 1, 2, ..., D-1$$

**代码实现**:
```python
def alr_transform(data, reference_column):
    """
    加法对数比转换
    
    参数:
    - data: 成分数据
    - reference_column: 参考列名或索引
    
    返回:
    - alr_data: ALR转换后的数据
    """
    if isinstance(data, pd.DataFrame):
        reference_data = data[reference_column].values
        other_columns = [col for col in data.columns if col != reference_column]
        other_data = data[other_columns].values
    else:
        reference_data = data[:, reference_column]
        other_data = np.delete(data, reference_column, axis=1)
    
    # 添加小常数避免对数零
    epsilon = 1e-10
    reference_data = reference_data + epsilon
    other_data = other_data + epsilon
    
    # ALR转换
    alr_data = np.log(other_data / reference_data[:, np.newaxis])
    
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(alr_data, index=data.index, columns=other_columns)
    else:
        return alr_data
```

#### 2.3 等距对数比转换 (ILR)

**数学原理**:
使用正交基向量进行转换：

$$ilr(\mathbf{x}) = \mathbf{V}^T \cdot clr(\mathbf{x})$$

其中 $\mathbf{V}$ 是正交矩阵。

**代码实现**:
```python
def ilr_transform(data):
    """
    等距对数比转换
    
    参数:
    - data: 成分数据
    
    返回:
    - ilr_data: ILR转换后的数据
    """
    from scipy.linalg import orth
    
    # 首先进行CLR转换
    clr_data = clr_transform(data)
    
    # 创建正交基
    if isinstance(clr_data, pd.DataFrame):
        n_components = clr_data.shape[1]
        V = orth(np.random.randn(n_components, n_components - 1))
    else:
        n_components = clr_data.shape[1]
        V = orth(np.random.randn(n_components, n_components - 1))
    
    # ILR转换
    ilr_data = clr_data @ V
    
    if isinstance(data, pd.DataFrame):
        column_names = [f'ILR_{i+1}' for i in range(ilr_data.shape[1])]
        return pd.DataFrame(ilr_data, index=data.index, columns=column_names)
    else:
        return ilr_data
```

## 🌊 分形异常检测算法

### 1. C-A分形模型

#### 1.1 浓度-面积关系计算

**数学原理**:
对于给定浓度阈值 $c$，计算浓度大于等于 $c$ 的面积 $A(c)$：

$$A(c) = \text{Area}\{x \in \Omega | C(x) \geq c\}$$

**代码实现**:
```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_ca_relationship(data, n_bins=100):
    """
    计算浓度-面积关系
    
    参数:
    - data: 地球化学数据
    - n_bins: 浓度区间数量
    
    返回:
    - concentrations: 浓度数组
    - areas: 面积数组
    """
    # 创建浓度区间
    concentrations = np.linspace(data.min(), data.max(), n_bins)
    areas = []
    
    for c in concentrations:
        # 计算浓度大于等于c的面积（样品数）
        area = np.sum(data >= c)
        areas.append(area)
    
    return concentrations, np.array(areas)

def plot_ca_loglog(concentrations, areas, title="C-A分形关系"):
    """
    绘制C-A双对数图
    
    参数:
    - concentrations: 浓度数组
    - areas: 面积数组
    - title: 图表标题
    """
    # 过滤零值
    valid_mask = (concentrations > 0) & (areas > 0)
    log_c = np.log10(concentrations[valid_mask])
    log_a = np.log10(areas[valid_mask])
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_c, log_a, 'b-', linewidth=2, label='C-A关系')
    plt.xlabel('log(浓度)')
    plt.ylabel('log(面积)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return log_c, log_a
```

#### 1.2 分形断点检测

**数学原理**:
使用一阶导数检测C-A曲线的断点：

$$\frac{d\log A}{d\log C} = \frac{\Delta \log A}{\Delta \log C}$$

**代码实现**:
```python
from scipy.signal import find_peaks
from scipy.stats import linregress

def detect_fractal_breaks(log_c, log_a, min_distance=5):
    """
    检测分形断点
    
    参数:
    - log_c: 对数浓度
    - log_a: 对数面积
    - min_distance: 最小断点间距
    
    返回:
    - break_points: 断点索引
    - derivatives: 导数数组
    - fractal_dimensions: 各段分形维数
    """
    # 计算一阶导数
    derivatives = np.diff(log_a) / np.diff(log_c)
    
    # 寻找导数的极值点（断点）
    peaks, _ = find_peaks(-np.abs(derivatives), distance=min_distance)
    
    # 计算各段的分形维数
    fractal_dimensions = []
    
    # 添加起始和结束点
    all_points = [0] + list(peaks) + [len(log_c) - 1]
    
    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i + 1]
        
        if end_idx - start_idx > 1:  # 至少需要2个点
            slope, _, _, _, _ = linregress(
                log_c[start_idx:end_idx], 
                log_a[start_idx:end_idx]
            )
            fractal_dimension = -slope
            fractal_dimensions.append(fractal_dimension)
        else:
            fractal_dimensions.append(None)
    
    return peaks, derivatives, fractal_dimensions, all_points
```

#### 1.3 阈值计算方法

##### 1.3.1 膝点检测法

**数学原理**:
寻找C-A曲线的最大曲率点：

$$\kappa = \frac{|y''|}{(1 + y'^2)^{3/2}}$$

**代码实现**:
```python
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

def knee_detection_threshold(concentrations, areas):
    """
    使用膝点检测法计算阈值
    
    参数:
    - concentrations: 浓度数组
    - areas: 面积数组
    
    返回:
    - threshold: 异常阈值
    - knee_point: 膝点索引
    """
    # 过滤有效数据
    valid_mask = (concentrations > 0) & (areas > 0)
    log_c = np.log10(concentrations[valid_mask])
    log_a = np.log10(areas[valid_mask])
    
    # 计算曲率
    first_derivative = np.gradient(log_a, log_c)
    second_derivative = np.gradient(first_derivative, log_c)
    
    curvature = np.abs(second_derivative) / (1 + first_derivative**2)**1.5
    
    # 寻找最大曲率点
    knee_point = np.argmax(curvature)
    
    # 计算阈值
    threshold = concentrations[valid_mask][knee_point]
    
    return threshold, knee_point
```

##### 1.3.2 K-means聚类法

**数学原理**:
使用K-means将C-A数据分为两类，寻找分类边界。

**代码实现**:
```python
from sklearn.cluster import KMeans

def kmeans_threshold(concentrations, areas, n_clusters=2):
    """
    使用K-means聚类法计算阈值
    
    参数:
    - concentrations: 浓度数组
    - areas: 面积数组
    - n_clusters: 聚类数量
    
    返回:
    - threshold: 异常阈值
    - labels: 聚类标签
    """
    # 准备数据
    valid_mask = (concentrations > 0) & (areas > 0)
    log_c = np.log10(concentrations[valid_mask])
    log_a = np.log10(areas[valid_mask])
    
    # 组合特征
    features = np.column_stack([log_c, log_a])
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # 找到异常类（通常浓度较高）
    cluster_centers = kmeans.cluster_centers_
    anomaly_cluster = np.argmax(cluster_centers[:, 0])  # 浓度最高的类
    
    # 计算阈值（异常类的最小浓度）
    anomaly_mask = labels == anomaly_cluster
    threshold_idx = np.where(valid_mask)[0][anomaly_mask].min()
    threshold = concentrations[threshold_idx]
    
    return threshold, labels
```

##### 1.3.3 分段线性拟合法

**数学原理**:
将C-A曲线分为两段线性部分，寻找最优分割点。

**代码实现**:
```python
def piecewise_linear_threshold(concentrations, areas):
    """
    使用分段线性拟合法计算阈值
    
    参数:
    - concentrations: 浓度数组
    - areas: 面积数组
    
    返回:
    - threshold: 异常阈值
    - break_point: 断点索引
    - r_squared: 拟合优度
    """
    # 过滤有效数据
    valid_mask = (concentrations > 0) & (areas > 0)
    log_c = np.log10(concentrations[valid_mask])
    log_a = np.log10(areas[valid_mask])
    
    best_r_squared = -np.inf
    best_break_point = None
    best_threshold = None
    
    # 尝试不同的断点位置
    for break_point in range(10, len(log_c) - 10):
        # 第一段拟合
        slope1, intercept1, _, _, _ = linregress(
            log_c[:break_point], log_a[:break_point]
        )
        
        # 第二段拟合
        slope2, intercept2, _, _, _ = linregress(
            log_c[break_point:], log_a[break_point:]
        )
        
        # 计算预测值
        pred1 = slope1 * log_c[:break_point] + intercept1
        pred2 = slope2 * log_c[break_point:] + intercept2
        pred_all = np.concatenate([pred1, pred2])
        
        # 计算R²
        ss_res = np.sum((log_a - pred_all) ** 2)
        ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_break_point = break_point
            best_threshold = concentrations[valid_mask][break_point]
    
    return best_threshold, best_break_point, best_r_squared
```

## ⚖️ 证据权算法

### 1. 基础证据权计算

#### 1.1 权重计算公式

**数学原理**:
对于证据图层 $E$ 和目标图层 $T$：

$$W^+ = \ln\left(\frac{P(E|T)}{P(E|\bar{T})}\right) = \ln\left(\frac{N(E \cap T)/N(T)}{N(E \cap \bar{T})/N(\bar{T})}\right)$$

$$W^- = \ln\left(\frac{P(\bar{E}|T)}{P(\bar{E}|\bar{T})}\right) = \ln\left(\frac{N(\bar{E} \cap T)/N(T)}{N(\bar{E} \cap \bar{T})/N(\bar{T})}\right)$$

$$C = W^+ - W^-$$

其中：
- $W^+$: 证据存在时的权重
- $W^-$: 证据不存在时的权重
- $C$: 对比度
- $N(\cdot)$: 单元格数量

**代码实现**:
```python
import numpy as np
from scipy.stats import norm

def calculate_weights(evidence_map, target_map):
    """
    计算证据权
    
    参数:
    - evidence_map: 证据图层 (二值数组)
    - target_map: 目标图层 (二值数组)
    
    返回:
    - weights: 权重字典
    """
    # 计算各种统计量
    total_cells = np.prod(evidence_map.shape)
    target_cells = np.sum(target_map > 0)
    non_target_cells = total_cells - target_cells
    
    evidence_with_target = np.sum((evidence_map > 0) & (target_map > 0))
    evidence_without_target = np.sum((evidence_map > 0) & (target_map == 0))
    
    no_evidence_with_target = np.sum((evidence_map == 0) & (target_map > 0))
    no_evidence_without_target = np.sum((evidence_map == 0) & (target_map == 0))
    
    # 计算权重
    w_plus = np.log((evidence_with_target / target_cells) / 
                   (evidence_without_target / non_target_cells))
    
    w_minus = np.log((no_evidence_with_target / target_cells) / 
                    (no_evidence_without_target / non_target_cells))
    
    contrast = w_plus - w_minus
    
    # 计算置信度
    s2_w_plus = (1 / evidence_with_target) + (1 / evidence_without_target)
    s2_w_minus = (1 / no_evidence_with_target) + (1 / no_evidence_without_target)
    s2_contrast = s2_w_plus + s2_w_minus
    
    studentized_contrast = contrast / np.sqrt(s2_contrast)
    
    return {
        'w_plus': w_plus,
        'w_minus': w_minus,
        'contrast': contrast,
        'studentized_contrast': studentized_contrast,
        's2_w_plus': s2_w_plus,
        's2_w_minus': s2_w_minus,
        's2_contrast': s2_contrast,
        'statistics': {
            'evidence_with_target': evidence_with_target,
            'evidence_without_target': evidence_without_target,
            'no_evidence_with_target': no_evidence_with_target,
            'no_evidence_without_target': no_evidence_without_target,
            'target_cells': target_cells,
            'non_target_cells': non_target_cells
        }
    }
```

#### 1.2 连续证据权重计算

**数学原理**:
对于连续证据，使用模糊隶属度函数转换为权重：

$$W(x) = W^+ \cdot \mu(x) + W^- \cdot (1 - \mu(x))$$

其中 $\mu(x)$ 是隶属度函数。

**代码实现**:
```python
def calculate_continuous_weights(evidence_data, target_data, 
                                membership_function='linear'):
    """
    计算连续证据权重
    
    参数:
    - evidence_data: 连续证据数据
    - target_data: 目标数据
    - membership_function: 隶属度函数类型
    
    返回:
    - weights: 权重字典
    """
    # 首先二值化以计算基础权重
    threshold = np.percentile(evidence_data, 80)  # 使用80%分位数作为阈值
    binary_evidence = (evidence_data >= threshold).astype(int)
    
    # 计算基础权重
    basic_weights = calculate_weights(binary_evidence, target_data)
    
    # 计算隶属度
    if membership_function == 'linear':
        membership = linear_membership(evidence_data, evidence_data.min(), evidence_data.max())
    elif membership_function == 'sigmoid':
        membership = sigmoid_membership(evidence_data)
    elif membership_function == 'gaussian':
        membership = gaussian_membership(evidence_data)
    else:
        raise ValueError(f"未知的隶属度函数: {membership_function}")
    
    # 计算连续权重
    continuous_weights = (basic_weights['w_plus'] * membership + 
                        basic_weights['w_minus'] * (1 - membership))
    
    return {
        'continuous_weights': continuous_weights,
        'membership': membership,
        'basic_weights': basic_weights
    }

def linear_membership(data, min_val, max_val):
    """线性隶属度函数"""
    return (data - min_val) / (max_val - min_val)

def sigmoid_membership(data, k=1, x0=0):
    """S型隶属度函数"""
    return 1 / (1 + np.exp(-k * (data - x0)))

def gaussian_membership(data, sigma=1):
    """高斯隶属度函数"""
    mean = np.mean(data)
    return np.exp(-0.5 * ((data - mean) / sigma) ** 2)
```

### 2. 条件独立性检验

#### 2.1 卡方检验

**数学原理**:
使用卡方检验验证证据间的条件独立性：

$$\chi^2 = \sum_{i,j,k} \frac{(O_{ijk} - E_{ijk})^2}{E_{ijk}}$$

其中 $O_{ijk}$ 是观测频数，$E_{ijk}$ 是期望频数。

**代码实现**:
```python
from scipy.stats import chi2_contingency

def test_conditional_independence(evidence1, evidence2, target):
    """
    检验条件独立性
    
    参数:
    - evidence1: 第一个证据图层
    - evidence2: 第二个证据图层
    - target: 目标图层
    
    返回:
    - test_result: 检验结果
    """
    # 创建三维列联表
    contingency_table = np.zeros((2, 2, 2))
    
    for i in [0, 1]:  # evidence1
        for j in [0, 1]:  # evidence2
            for k in [0, 1]:  # target
                mask = (evidence1 == i) & (evidence2 == j) & (target == k)
                contingency_table[i, j, k] = np.sum(mask)
    
    # 重塑为2D表格进行卡方检验
    contingency_2d = contingency_table.reshape(4, 2)
    
    # 执行卡方检验
    chi2, p_value, dof, expected = chi2_contingency(contingency_2d)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'contingency_table': contingency_table,
        'independent': p_value > 0.05  # 显著性水平0.05
    }
```

#### 2.2 互信息检验

**数学原理**:
使用互信息度量变量间的依赖关系：

$$I(X;Y) = \sum_{x,y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

**代码实现**:
```python
from sklearn.metrics import mutual_info_score

def mutual_information_test(evidence1, evidence2, target):
    """
    使用互信息检验条件独立性
    
    参数:
    - evidence1: 第一个证据图层
    - evidence2: 第二个证据图层
    - target: 目标图层
    
    返回:
    - mi_result: 互信息结果
    """
    # 计算互信息
    mi_e1_e2 = mutual_info_score(evidence1, evidence2)
    mi_e1_target = mutual_info_score(evidence1, target)
    mi_e2_target = mutual_info_score(evidence2, target)
    
    # 条件互信息近似
    conditional_mi = mi_e1_e2 - mi_e1_target - mi_e2_target
    
    return {
        'mi_e1_e2': mi_e1_e2,
        'mi_e1_target': mi_e1_target,
        'mi_e2_target': mi_e2_target,
        'conditional_mi': conditional_mi,
        'independent': conditional_mi < 0.1  # 阈值可调
    }
```

## 🤖 机器学习算法

### 1. 随机森林算法

#### 1.1 算法原理

**数学原理**:
随机森林通过集成多个决策树来提高预测性能：

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

其中 $T_b(x)$ 是第 $b$ 棵决策树的预测。

**代码实现**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

class CustomRandomForest:
    """自定义随机森林实现"""
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """训练随机森林"""
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_importances_ = np.zeros(n_features)
        
        # 确定每棵树的特征数量
        if self.max_features == 'sqrt':
            n_features_per_tree = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_features_per_tree = int(np.log2(n_features))
        else:
            n_features_per_tree = self.max_features
        
        np.random.seed(self.random_state)
        
        for _ in range(self.n_estimators):
            # Bootstrap采样
            bootstrap_indices = np.random.choice(
                n_samples, n_samples, replace=True
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # 随机选择特征
            feature_indices = np.random.choice(
                n_features, n_features_per_tree, replace=False
            )
            
            # 训练决策树
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            
            tree.fit(X_bootstrap[:, feature_indices], y_bootstrap)
            
            # 保存树和特征索引
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
            
            # 累积特征重要性
            tree_importance = np.zeros(n_features)
            tree_importance[feature_indices] = tree.feature_importances_
            self.feature_importances_ += tree_importance
        
        # 平均特征重要性
        self.feature_importances_ /= self.n_estimators
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            tree_pred = tree.predict_proba(X[:, feature_indices])
            predictions.append(tree_pred)
        
        # 平均预测
        avg_predictions = np.mean(predictions, axis=0)
        return avg_predictions
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
```

### 2. 支持向量机算法

#### 2.1 算法原理

**数学原理**:
SVM通过寻找最优超平面来分类数据：

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i$$

约束条件：
$$y_i(w^T\phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**代码实现**:
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVMProspectivityModel:
    """SVM找矿预测模型"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', 
                 probability=True, random_state=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.svm = None
    
    def fit(self, X, y):
        """训练SVM模型"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建SVM模型
        self.svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state
        )
        
        # 训练模型
        self.svm.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        if self.svm is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)
    
    def predict(self, X):
        """预测类别"""
        if self.svm is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)
    
    def get_support_vectors(self):
        """获取支持向量"""
        if self.svm is None:
            raise ValueError("模型尚未训练")
        
        return self.svm.support_vectors_
    
    def decision_function(self, X):
        """决策函数值"""
        if self.svm is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        return self.svm.decision_function(X_scaled)
```

### 3. 神经网络算法

#### 3.1 多层感知机

**数学原理**:
前向传播：
$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

反向传播：
$$\delta^{(l)} = (W^{(l+1)})^T\delta^{(l+1)} \odot \sigma'(z^{(l)})$$

**代码实现**:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class NeuralNetworkProspectivityModel:
    """神经网络找矿预测模型"""
    
    def __init__(self, hidden_layers=[64, 32, 16], 
                 activation='relu', dropout_rate=0.3,
                 learning_rate=0.001, random_state=None):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self, input_dim):
        """构建神经网络模型"""
        tf.random.set_seed(self.random_state)
        
        model = Sequential()
        
        # 输入层
        model.add(Dense(self.hidden_layers[0], 
                       input_dim=input_dim, 
                       activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # 隐藏层
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # 输出层
        model.add(Dense(1, activation='sigmoid'))
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def fit(self, X, y, validation_split=0.2, epochs=100, 
            batch_size=32, verbose=1):
        """训练神经网络模型"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        self.model = self.build_model(X.shape[1])
        
        # 早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 训练模型
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return history
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # 返回二分类概率
        return np.column_stack([1 - predictions, predictions])
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self, X, y):
        """获取特征重要性（使用排列重要性）"""
        from sklearn.inspection import permutation_importance
        
        # 训练模型（如果尚未训练）
        if self.model is None:
            self.fit(X, y, verbose=0)
        
        # 计算排列重要性
        X_scaled = self.scaler.transform(X)
        
        def model_predict(X):
            return self.model.predict(X).flatten()
        
        result = permutation_importance(
            model_predict, X_scaled, y,
            n_repeats=10, random_state=self.random_state
        )
        
        return result.importances_mean
```

## 📈 模型评估算法

### 1. 交叉验证算法

#### 1.1 空间交叉验证

**数学原理**:
空间交叉验证考虑空间自相关性，避免空间过拟合：

$$CV = \frac{1}{K}\sum_{k=1}^{K} \text{Score}(f_{-k}, D_k)$$

其中 $f_{-k}$ 是在第 $k$ 折之外的数据上训练的模型。

**代码实现**:
```python
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import numpy as np

class SpatialCrossValidation:
    """空间交叉验证"""
    
    def __init__(self, n_splits=5, spatial_cv=True, random_state=None):
        self.n_splits = n_splits
        self.spatial_cv = spatial_cv
        self.random_state = random_state
    
    def split(self, X, y, coordinates):
        """生成空间交叉验证分割"""
        if self.spatial_cv:
            return self._spatial_split(coordinates)
        else:
            return self._random_split(len(X))
    
    def _spatial_split(self, coordinates):
        """空间分割"""
        # 使用K-means聚类分割空间
        kmeans = KMeans(n_clusters=self.n_splits, 
                       random_state=self.random_state)
        spatial_clusters = kmeans.fit_predict(coordinates)
        
        splits = []
        for i in range(self.n_splits):
            train_mask = spatial_clusters != i
            test_mask = spatial_clusters == i
            splits.append((train_mask, test_mask))
        
        return splits
    
    def _random_split(self, n_samples):
        """随机分割"""
        kf = KFold(n_splits=self.n_splits, 
                  shuffle=True, 
                  random_state=self.random_state)
        return kf.split(np.arange(n_samples))
    
    def evaluate(self, model, X, y, coordinates, scoring='roc_auc'):
        """评估模型"""
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        scores = []
        splits = self.split(X, y, coordinates)
        
        for train_mask, test_mask in splits:
            # 训练模型
            model.fit(X[train_mask], y[train_mask])
            
            # 预测
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X[test_mask])[:, 1]
            else:
                y_pred = model.predict(X[test_mask])
            
            # 计算评分
            if scoring == 'roc_auc':
                score = roc_auc_score(y[test_mask], y_pred)
            elif scoring == 'accuracy':
                y_pred_class = (y_pred > 0.5).astype(int)
                score = accuracy_score(y[test_mask], y_pred_class)
            elif scoring == 'f1':
                y_pred_class = (y_pred > 0.5).astype(int)
                score = f1_score(y[test_mask], y_pred_class)
            else:
                raise ValueError(f"未知的评分指标: {scoring}")
            
            scores.append(score)
        
        return np.array(scores)
```

### 2. 成功率曲线算法

#### 2.1 成功率计算

**数学原理**:
成功率曲线衡量预测模型在不同面积比例下的预测效果：

$$SR(A) = \frac{\text{目标区域在预测前}A\%\text{区域中的比例}}{A\%}$$

**代码实现**:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

class SuccessRateCurve:
    """成功率曲线分析"""
    
    def __init__(self):
        self.area_percentages = None
        self.success_rates = None
        self.auc_score = None
    
    def calculate(self, predictions, targets, area_percentages=None):
        """
        计算成功率曲线
        
        参数:
        - predictions: 预测概率
        - targets: 真实标签
        - area_percentages: 面积百分比数组
        
        返回:
        - area_percentages: 面积百分比
        - success_rates: 成功率
        - auc_score: AUC分数
        """
        if area_percentages is None:
            area_percentages = np.arange(1, 101, 1)
        
        success_rates = []
        
        for area_pct in area_percentages:
            # 选择前area_pct%的预测值
            threshold = np.percentile(predictions, 100 - area_pct)
            selected_mask = predictions >= threshold
            
            # 计算成功率
            if np.sum(selected_mask) > 0:
                success_rate = np.sum(targets[selected_mask]) / np.sum(selected_mask)
            else:
                success_rate = 0
            
            success_rates.append(success_rate)
        
        # 计算AUC
        auc_score = auc(area_percentages / 100, success_rates)
        
        self.area_percentages = area_percentages
        self.success_rates = np.array(success_rates)
        self.auc_score = auc_score
        
        return area_percentages, np.array(success_rates), auc_score
    
    def plot(self, title="成功率曲线"):
        """绘制成功率曲线"""
        if self.area_percentages is None or self.success_rates is None:
            raise ValueError("请先调用calculate方法")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.area_percentages, self.success_rates, 
                'b-', linewidth=2, label=f'预测模型 (AUC={self.auc_score:.3f})')
        
        # 随机预测基线
        random_line = self.area_percentages / 100
        plt.plot(self.area_percentages, random_line, 
                'r--', linewidth=2, label='随机预测')
        
        plt.xlabel('预测区域面积 (%)')
        plt.ylabel('成功率')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 1)
        plt.show()
    
    def get_optimal_threshold(self, predictions, targets):
        """获取最优阈值"""
        if self.area_percentages is None or self.success_rates is None:
            self.calculate(predictions, targets)
        
        # 找到成功率最高的点
        optimal_idx = np.argmax(self.success_rates)
        optimal_area_pct = self.area_percentages[optimal_idx]
        optimal_threshold = np.percentile(predictions, 100 - optimal_area_pct)
        
        return {
            'threshold': optimal_threshold,
            'area_percentage': optimal_area_pct,
            'success_rate': self.success_rates[optimal_idx]
        }
```

## 📚 总结

本文档详细介绍了Gold-Seeker平台中使用的各种算法，包括：

1. **地球化学数据处理**: 删失数据处理、数据转换算法
2. **分形异常检测**: C-A分形模型、阈值计算方法
3. **证据权方法**: 权重计算、条件独立性检验
4. **机器学习**: 随机森林、支持向量机、神经网络
5. **模型评估**: 交叉验证、成功率曲线

这些算法构成了Gold-Seeker平台的技术基础，为地球化学找矿预测提供了科学、可靠的方法支撑。每种算法都经过精心设计和实现，确保计算结果的准确性和可靠性。