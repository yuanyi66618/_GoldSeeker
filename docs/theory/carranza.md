# Carranza理论基础

本文档详细介绍Emmanuel John M. Carranza的地球化学异常与矿产远景制图理论，这是Gold-Seeker平台的核心理论基础。

## 📚 理论概述

### Carranza (2009) 核心理论

Carranza在《Geochemical Anomaly and Mineral Prospectivity Mapping in GIS》一书中提出了系统的地球化学找矿预测理论框架，该理论已成为现代地球化学勘探的标准方法。

#### 理论核心要素

1. **地球化学异常识别**: 基于统计学和分形理论的异常检测方法
2. **证据图层构建**: 将地球化学数据转换为找矿证据
3. **证据权方法**: 定量评估各证据图层的重要性
4. **空间分析**: 结合地质、地球化学和遥感数据进行综合分析
5. **GIS集成**: 在地理信息系统中进行空间建模和可视化

### 理论发展历程

```
传统地球化学勘探 → 统计学方法 → 分形理论 → GIS集成 → 多智能体系统
     (1970s)           (1980s)      (1990s)      (2000s)        (2020s)
```

## 🔬 地球化学异常理论

### 异常定义

#### 背景与异常
```python
# 背景与异常的数学定义
def classify_anomaly(data, threshold):
    """
    根据阈值分类背景和异常
    
    参数:
    - data: 地球化学数据
    - threshold: 异常阈值
    
    返回:
    - background: 背景值
    - anomaly: 异常值
    """
    background = data[data <= threshold]
    anomaly = data[data > threshold]
    return background, anomaly
```

#### 异常类型

1. **局部异常**: 局部高值区域，通常与矿化直接相关
2. **区域异常**: 大范围的高值区域，可能反映地质构造
3. **多重异常**: 多个异常区域的组合，指示复杂的成矿过程

### 统计学异常识别

#### 传统统计方法

```python
class StatisticalAnomalyDetector:
    """基于统计学的异常检测"""
    
    def mean_plus_2sd(self, data):
        """均值+2倍标准差法"""
        mean = np.mean(data)
        std = np.std(data)
        threshold = mean + 2 * std
        return threshold
    
    def percentile_method(self, data, percentile=95):
        """百分位数法"""
        threshold = np.percentile(data, percentile)
        return threshold
    
    def boxplot_method(self, data):
        """箱线图法"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        return threshold
```

#### 多元统计方法

```python
class MultivariateAnomalyDetector:
    """多元统计异常检测"""
    
    def mahalanobis_distance(self, data, cov_inv=None):
        """马氏距离法"""
        if cov_inv is None:
            cov = np.cov(data.T)
            cov_inv = np.linalg.inv(cov)
        
        mean = np.mean(data, axis=0)
        diff = data - mean
        mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        return mahal_dist
    
    def principal_component_analysis(self, data, n_components=2):
        """主成分分析"""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return transformed, pca.explained_variance_ratio_
```

## 🌊 分形异常理论

### 分形理论基础

#### C-A分形模型

Cheng, Agterberg和Ballantyne (1994) 提出的浓度-面积（Concentration-Area, C-A）分形模型是Carranza理论的重要组成部分。

```python
class FractalAnomalyDetector:
    """基于分形理论的异常检测"""
    
    def calculate_ca_relationship(self, data, bins=100):
        """计算C-A关系"""
        # 创建浓度区间
        concentrations = np.linspace(data.min(), data.max(), bins)
        areas = []
        
        for c in concentrations:
            area = np.sum(data >= c)
            areas.append(area)
        
        return concentrations, areas
    
    def plot_ca_loglog(self, concentrations, areas):
        """绘制C-A双对数图"""
        log_c = np.log10(concentrations[concentrations > 0])
        log_a = np.log10(areas[concentrations > 0])
        
        plt.figure(figsize=(10, 6))
        plt.plot(log_c, log_a, 'b-', linewidth=2)
        plt.xlabel('log(浓度)')
        plt.ylabel('log(面积)')
        plt.title('C-A分形关系')
        plt.grid(True)
        plt.show()
        
        return log_c, log_a
    
    def detect_fractal_breaks(self, log_c, log_a):
        """检测分形断点"""
        # 使用一阶导数检测断点
        derivatives = np.diff(log_a) / np.diff(log_c)
        
        # 寻找导数的极值点
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-np.abs(derivatives), distance=5)
        
        return peaks, derivatives
```

#### 分形维数计算

```python
def calculate_fractal_dimension(log_c, log_a, start_idx, end_idx):
    """计算分形维数"""
    # 线性回归拟合
    x = log_c[start_idx:end_idx]
    y = log_a[start_idx:end_idx]
    
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    
    # 分形维数是斜率的负值
    fractal_dimension = -slope
    
    return fractal_dimension, coeffs
```

### 多重分形分析

#### 多重分形谱

```python
class MultifractalAnalysis:
    """多重分形分析"""
    
    def calculate_multifractal_spectrum(self, data, q_values=None):
        """计算多重分形谱"""
        if q_values is None:
            q_values = np.linspace(-5, 5, 21)
        
        tau_q = []
        alpha_q = []
        f_alpha = []
        
        for q in q_values:
            # 计算配分函数
            partition = self.calculate_partition_function(data, q)
            tau_q.append(np.log(partition))
            
            # 计算奇异指数
            alpha = self.calculate_singularity_exponent(data, q)
            alpha_q.append(alpha)
            
            # 计算多重分形谱
            f_alpha.append(q * alpha - tau_q[-1])
        
        return q_values, tau_q, alpha_q, f_alpha
    
    def calculate_partition_function(self, data, q, scales=None):
        """计算配分函数"""
        if scales is None:
            scales = [2, 4, 8, 16, 32]
        
        partition_values = []
        
        for scale in scales:
            # 将数据分割为尺度为scale的盒子
            boxes = self.partition_data(data, scale)
            
            # 计算每个盒子的概率
            probabilities = [np.sum(box) / np.sum(data) for box in boxes]
            
            # 计算配分函数
            partition = np.sum([p**q for p in probabilities if p > 0])
            partition_values.append(partition)
        
        return partition_values
```

## ⚖️ 证据权方法

### 证据权理论基础

#### 基本概念

证据权方法（Weights of Evidence, WofE）是一种基于贝叶斯定理的定量预测方法，用于评估各种证据图层对目标矿床的指示作用。

```python
class WeightsOfEvidence:
    """证据权方法实现"""
    
    def calculate_weights(self, evidence_map, target_map):
        """计算证据权"""
        # 计算各种统计量
        total_area = np.prod(evidence_map.shape)
        target_area = np.sum(target_map > 0)
        
        # 计算W+（证据存在时的权重）
        evidence_with_target = np.sum((evidence_map > 0) & (target_map > 0))
        evidence_without_target = np.sum((evidence_map > 0) & (target_map == 0))
        
        w_plus = np.log((evidence_with_target / target_area) / 
                       (evidence_without_target / (total_area - target_area)))
        
        # 计算W-（证据不存在时的权重）
        no_evidence_with_target = np.sum((evidence_map == 0) & (target_map > 0))
        no_evidence_without_target = np.sum((evidence_map == 0) & (target_map == 0))
        
        w_minus = np.log((no_evidence_with_target / target_area) / 
                        (no_evidence_without_target / (total_area - target_area)))
        
        # 计算对比度
        contrast = w_plus - w_minus
        
        return {
            'w_plus': w_plus,
            'w_minus': w_minus,
            'contrast': contrast
        }
    
    def calculate_studentized_contrast(self, w_plus, w_minus, n_plus, n_minus):
        """计算学生化对比度"""
        contrast = w_plus - w_minus
        
        # 计算方差
        var_w_plus = 1 / n_plus + 1 / n_minus
        var_w_minus = 1 / n_plus + 1 / n_minus
        
        # 计算学生化对比度
        studentized_contrast = contrast / np.sqrt(var_w_plus + var_w_minus)
        
        return studentized_contrast
```

#### 证据组合

```python
def combine_evidence(weights_list):
    """组合多个证据的权重"""
    total_w_plus = sum([w['w_plus'] for w in weights_list])
    total_w_minus = sum([w['w_minus'] for w in weights_list])
    total_contrast = sum([w['contrast'] for w in weights_list])
    
    return {
        'total_w_plus': total_w_plus,
        'total_w_minus': total_w_minus,
        'total_contrast': total_contrast
    }
```

### 条件独立性检验

#### 卡方检验

```python
def chi_square_test(evidence1, evidence2, target):
    """卡方检验条件独立性"""
    # 创建列联表
    contingency_table = np.zeros((2, 2, 2))
    
    # 填充列联表
    for i in [0, 1]:  # evidence1
        for j in [0, 1]:  # evidence2
            for k in [0, 1]:  # target
                mask = (evidence1 == i) & (evidence2 == j) & (target == k)
                contingency_table[i, j, k] = np.sum(mask)
    
    # 执行卡方检验
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(contingency_table.reshape(4, 2))
    
    return chi2, p_value, dof
```

## 🗺️ 空间分析理论

### 空间自相关

#### Moran's I

```python
def calculate_morans_i(data, weights_matrix):
    """计算Moran's I空间自相关指数"""
    n = len(data)
    
    # 计算均值
    mean_data = np.mean(data)
    
    # 计算分子
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights_matrix[i, j] * (data[i] - mean_data) * (data[j] - mean_data)
    
    # 计算分母
    denominator = np.sum((data - mean_data) ** 2)
    
    # 计算权重总和
    sum_weights = np.sum(weights_matrix)
    
    # 计算Moran's I
    morans_i = (n / sum_weights) * (numerator / denominator)
    
    return morans_i
```

#### Getis-Ord G*

```python
def calculate_getis_ord_g(data, coordinates, distance_threshold):
    """计算Getis-Ord G*统计量"""
    n = len(data)
    g_star_values = np.zeros(n)
    
    # 计算距离矩阵
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(coordinates, coordinates)
    
    # 创建权重矩阵
    weights_matrix = (distance_matrix <= distance_threshold).astype(float)
    np.fill_diagonal(weights_matrix, 0)  # 排除自身
    
    for i in range(n):
        # 获取邻居
        neighbors = np.where(weights_matrix[i] > 0)[0]
        
        if len(neighbors) > 0:
            # 计算G*值
            sum_data = np.sum(data[neighbors])
            sum_weights = np.sum(weights_matrix[i, neighbors])
            
            g_star_values[i] = sum_data / sum_weights
    
    return g_star_values
```

### 空间插值

#### 克里金插值

```python
class KrigingInterpolator:
    """克里金插值器"""
    
    def __init__(self, variogram_model='spherical'):
        self.variogram_model = variogram_model
        self.fitted_model = None
    
    def fit_variogram(self, coordinates, values):
        """拟合变异函数"""
        from pykrige.ok import OrdinaryKriging
        
        # 创建普通克里金模型
        ok = OrdinaryKriging(
            coordinates[:, 0], coordinates[:, 1], values,
            variogram_model=self.variogram_model,
            verbose=False
        )
        
        self.fitted_model = ok
        return ok
    
    def interpolate(self, grid_x, grid_y):
        """执行插值"""
        if self.fitted_model is None:
            raise ValueError("必须先拟合变异函数")
        
        z, ss = self.fitted_model.execute('grid', grid_x, grid_y)
        return z, ss
```

## 📊 数据处理理论

### 数据预处理

#### 删失数据处理

```python
class CensoredDataProcessor:
    """删失数据处理器"""
    
    def __init__(self, detection_limits):
        self.detection_limits = detection_limits
    
    def impute_censored_data(self, data, method='dl_over_2'):
        """处理删失数据"""
        imputed_data = data.copy()
        
        for element, dl in self.detection_limits.items():
            if element in data.columns:
                censored_mask = data[element] < dl
                
                if method == 'dl_over_2':
                    # 检测限/2方法
                    imputed_data.loc[censored_mask, element] = dl / 2
                
                elif method == 'rosner':
                    # Rosner方法
                    imputed_data.loc[censored_mask, element] = self._rosner_imputation(
                        data[element], censored_mask, dl
                    )
                
                elif method == 'maximum_likelihood':
                    # 最大似然估计
                    imputed_data.loc[censored_mask, element] = self._ml_imputation(
                        data[element], censored_mask, dl
                    )
        
        return imputed_data
    
    def _rosner_imputation(self, data, censored_mask, dl):
        """Rosner删失数据插补"""
        from scipy import stats
        
        # 使用非删失数据估计参数
        uncensored_data = data[~censored_mask]
        mean, std = stats.norm.fit(uncensored_data)
        
        # 计算删失数据的期望值
        from scipy.stats import norm
        z_dl = (dl - mean) / std
        expected_value = mean - std * norm.pdf(z_dl) / norm.cdf(z_dl)
        
        return expected_value
```

#### 数据转换

```python
class DataTransformer:
    """数据转换器"""
    
    def clr_transform(self, data):
        """中心对数比转换"""
        # 添加小常数避免对数零
        epsilon = 1e-10
        data = data + epsilon
        
        # 计算几何均值
        geometric_mean = np.exp(np.mean(np.log(data), axis=1))
        
        # CLR转换
        clr_data = np.log(data.values / geometric_mean[:, np.newaxis])
        
        return clr_data
    
    def alr_transform(self, data, reference_column):
        """加法对数比转换"""
        reference_data = data[reference_column].values
        alr_data = np.log(data.drop(columns=[reference_column]).values / reference_data[:, np.newaxis])
        
        return alr_data
    
    def ilr_transform(self, data):
        """等距对数比转换"""
        from skbio.stats.composition import ilr
        ilr_data = ilr(data.values)
        
        return ilr_data
```

### 异常值检测

#### 多元异常值检测

```python
class MultivariateOutlierDetector:
    """多元异常值检测器"""
    
    def __init__(self, method='mahalanobis'):
        self.method = method
    
    def detect_outliers(self, data):
        """检测异常值"""
        if self.method == 'mahalanobis':
            return self._mahalanobis_detection(data)
        elif self.method == 'robust_mahalanobis':
            return self._robust_mahalanobis_detection(data)
        elif self.method == 'isolation_forest':
            return self._isolation_forest_detection(data)
        else:
            raise ValueError(f"未知方法: {self.method}")
    
    def _mahalanobis_detection(self, data):
        """马氏距离异常值检测"""
        from scipy.stats import chi2
        
        # 计算马氏距离
        cov_matrix = np.cov(data.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mean_vector = np.mean(data, axis=0)
        
        mahal_distances = []
        for i in range(len(data)):
            diff = data[i] - mean_vector
            mahal_dist = np.sqrt(diff @ inv_cov_matrix @ diff.T)
            mahal_distances.append(mahal_dist)
        
        # 计算阈值
        threshold = chi2.ppf(0.975, df=data.shape[1])
        
        # 标识异常值
        outliers = np.array(mahal_distances) > np.sqrt(threshold)
        
        return outliers, mahal_distances
```

## 🎯 找矿预测模型

### 证据图层构建

#### 二值证据图层

```python
class BinaryEvidenceLayer:
    """二值证据图层构建器"""
    
    def create_binary_layer(self, continuous_data, threshold, operator='>'):
        """创建二值证据图层"""
        if operator == '>':
            binary_layer = (continuous_data > threshold).astype(int)
        elif operator == '<':
            binary_layer = (continuous_data < threshold).astype(int)
        elif operator == '>=':
            binary_layer = (continuous_data >= threshold).astype(int)
        elif operator == '<=':
            binary_layer = (continuous_data <= threshold).astype(int)
        else:
            raise ValueError(f"未知操作符: {operator}")
        
        return binary_layer
    
    def optimize_threshold(self, evidence_data, target_data):
        """优化阈值选择"""
        thresholds = np.linspace(evidence_data.min(), evidence_data.max(), 100)
        best_threshold = None
        best_contrast = -np.inf
        
        for threshold in thresholds:
            binary_layer = self.create_binary_layer(evidence_data, threshold)
            weights = self.calculate_weights(binary_layer, target_data)
            
            if weights['contrast'] > best_contrast:
                best_contrast = weights['contrast']
                best_threshold = threshold
        
        return best_threshold, best_contrast
```

#### 连续证据图层

```python
class ContinuousEvidenceLayer:
    """连续证据图层构建器"""
    
    def fuzzy_membership(self, data, membership_type='linear'):
        """模糊隶属度转换"""
        if membership_type == 'linear':
            return self._linear_fuzzy(data)
        elif membership_type == 'sigmoid':
            return self._sigmoid_fuzzy(data)
        elif membership_type == 'gaussian':
            return self._gaussian_fuzzy(data)
        else:
            raise ValueError(f"未知隶属度类型: {membership_type}")
    
    def _linear_fuzzy(self, data):
        """线性模糊隶属度"""
        min_val = data.min()
        max_val = data.max()
        
        if max_val == min_val:
            return np.ones_like(data)
        
        return (data - min_val) / (max_val - min_val)
    
    def _sigmoid_fuzzy(self, data, k=1, x0=0):
        """S型模糊隶属度"""
        return 1 / (1 + np.exp(-k * (data - x0)))
```

### 预测模型集成

#### 模型集成方法

```python
class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, data):
        """集成预测"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(data)
            predictions.append(pred)
        
        # 加权平均
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def stacking_ensemble(self, train_data, train_target, test_data):
        """堆叠集成"""
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LinearRegression
        
        # 第一层预测
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((len(train_data), len(self.models)))
        
        for i, model in enumerate(self.models):
            fold_predictions = np.zeros(len(train_data))
            
            for train_idx, val_idx in kf.split(train_data):
                model.fit(train_data[train_idx], train_target[train_idx])
                fold_predictions[val_idx] = model.predict(train_data[val_idx])
            
            meta_features[:, i] = fold_predictions
        
        # 训练元模型
        meta_model = LinearRegression()
        meta_model.fit(meta_features, train_target)
        
        # 第二层预测
        test_meta_features = np.zeros((len(test_data), len(self.models)))
        for i, model in enumerate(self.models):
            model.fit(train_data, train_target)
            test_meta_features[:, i] = model.predict(test_data)
        
        final_prediction = meta_model.predict(test_meta_features)
        
        return final_prediction
```

## 📈 模型验证理论

### 交叉验证

#### 空间交叉验证

```python
class SpatialCrossValidation:
    """空间交叉验证"""
    
    def __init__(self, n_splits=5, spatial_cv=True):
        self.n_splits = n_splits
        self.spatial_cv = spatial_cv
    
    def spatial_split(self, coordinates, target):
        """空间分割"""
        if self.spatial_cv:
            return self._spatial_block_split(coordinates, target)
        else:
            return self._random_split(coordinates, target)
    
    def _spatial_block_split(self, coordinates, target):
        """空间块分割"""
        from sklearn.cluster import KMeans
        
        # 使用K-means聚类分割空间
        kmeans = KMeans(n_splits=self.n_splits, random_state=42)
        spatial_clusters = kmeans.fit_predict(coordinates)
        
        splits = []
        for i in range(self.n_splits):
            train_mask = spatial_clusters != i
            test_mask = spatial_clusters == i
            
            splits.append((train_mask, test_mask))
        
        return splits
    
    def evaluate_model(self, model, data, target, coordinates):
        """评估模型"""
        splits = self.spatial_split(coordinates, target)
        
        scores = []
        for train_mask, test_mask in splits:
            # 训练模型
            model.fit(data[train_mask], target[train_mask])
            
            # 预测
            predictions = model.predict(data[test_mask])
            
            # 计算评分
            score = self._calculate_score(target[test_mask], predictions)
            scores.append(score)
        
        return np.mean(scores), np.std(scores)
```

### 性能指标

#### 分类指标

```python
class ClassificationMetrics:
    """分类性能指标"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob=None):
        """计算所有分类指标"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1'] = f1_score(y_true, y_pred)
        
        if y_prob is not None:
            self.metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        self.metrics['classification_report'] = classification_report(y_true, y_pred)
        
        return self.metrics
    
    def calculate_success_rate(self, predictions, target_areas, total_area):
        """计算成功率曲线"""
        # 按预测值排序
        sorted_indices = np.argsort(predictions)[::-1]
        
        success_rates = []
        area_percentages = []
        
        for i in range(1, len(sorted_indices) + 1):
            top_indices = sorted_indices[:i]
            top_area = i / len(predictions) * 100
            
            # 计算包含的目标区域比例
            target_in_top = np.sum(target_areas[top_indices]) / np.sum(target_areas) * 100
            
            success_rates.append(target_in_top)
            area_percentages.append(top_area)
        
        return area_percentages, success_rates
```

## 📚 理论应用

### 卡林型金矿应用

#### 地球化学特征

```python
class CarlinTypeGoldAnalysis:
    """卡林型金矿地球化学分析"""
    
    def __init__(self):
        self.pathfinder_elements = ['Au', 'As', 'Sb', 'Hg', 'Tl', 'W']
        self.major_elements = ['Si', 'Al', 'Fe', 'Ca', 'Mg', 'Na', 'K']
    
    def identify_pathfinder_anomalies(self, geochemical_data):
        """识别路径元素异常"""
        anomalies = {}
        
        for element in self.pathfinder_elements:
            if element in geochemical_data.columns:
                # 使用分形方法检测异常
                detector = FractalAnomalyDetector()
                threshold = detector.calculate_fractal_threshold(
                    geochemical_data[element]
                )
                
                anomalies[element] = {
                    'threshold': threshold,
                    'anomaly_points': geochemical_data[geochemical_data[element] > threshold]
                }
        
        return anomalies
    
    def calculate_gold_association_index(self, geochemical_data):
        """计算金关联指数"""
        if 'Au' not in geochemical_data.columns:
            return None
        
        gold_correlations = {}
        for element in self.pathfinder_elements:
            if element != 'Au' and element in geochemical_data.columns:
                correlation = np.corrcoef(
                    geochemical_data['Au'], 
                    geochemical_data[element]
                )[0, 1]
                gold_correlations[element] = correlation
        
        return gold_correlations
```

### 斑岩型铜矿应用

#### 地球化学特征

```python
class PorphyryCopperAnalysis:
    """斑岩型铜矿地球化学分析"""
    
    def __init__(self):
        self.pathfinder_elements = ['Cu', 'Mo', 'Au', 'Ag', 'Re']
        self.alteration_elements = ['K', 'Na', 'Ca', 'Mg', 'Fe', 'Al']
    
    def identify_alteration_zones(self, geochemical_data):
        """识别蚀变带"""
        alteration_indices = {}
        
        # 钾化指数
        if 'K' in geochemical_data.columns and 'Na' in geochemical_data.columns:
            k_na_ratio = geochemical_data['K'] / geochemical_data['Na']
            alteration_indices['potassic'] = k_na_ratio
        
        # 青磐岩化指数
        if 'Ca' in geochemical_data.columns and 'Na' in geochemical_data.columns:
            ca_na_ratio = geochemical_data['Ca'] / geochemical_data['Na']
            alteration_indices['propylitic'] = ca_na_ratio
        
        # 泥化指数
        if 'Al' in geochemical_data.columns and 'K' in geochemical_data.columns:
            al_k_ratio = geochemical_data['Al'] / geochemical_data['K']
            alteration_indices['argillic'] = al_k_ratio
        
        return alteration_indices
    
    def calculate_copper_potential(self, geochemical_data):
        """计算铜矿潜力"""
        if 'Cu' not in geochemical_data.columns:
            return None
        
        # 多元素综合指数
        elements = ['Cu', 'Mo', 'Au']
        available_elements = [e for e in elements if e in geochemical_data.columns]
        
        if len(available_elements) < 2:
            return None
        
        # 标准化数据
        normalized_data = geochemical_data[available_elements].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        # 计算综合指数
        copper_potential = normalized_data.mean(axis=1)
        
        return copper_potential
```

## 🔮 理论发展

### 新兴理论方向

#### 机器学习集成

```python
class MLIntegratedGeochemistry:
    """机器学习集成的地球化学分析"""
    
    def __init__(self):
        self.traditional_methods = {
            'statistical': StatisticalAnomalyDetector(),
            'fractal': FractalAnomalyDetector(),
            'multivariate': MultivariateAnomalyDetector()
        }
        self.ml_methods = {
            'random_forest': None,
            'svm': None,
            'neural_network': None
        }
    
    def hybrid_anomaly_detection(self, data):
        """混合异常检测"""
        # 传统方法结果
        traditional_results = {}
        for method_name, method in self.traditional_methods.items():
            traditional_results[method_name] = method.detect_outliers(data)
        
        # 机器学习方法结果
        ml_results = {}
        for method_name, model in self.ml_methods.items():
            if model is not None:
                ml_results[method_name] = model.predict(data)
        
        # 集成结果
        ensemble_results = self._ensemble_results(
            traditional_results, ml_results
        )
        
        return ensemble_results
```

#### 深度学习应用

```python
class DeepLearningGeochemistry:
    """深度学习地球化学分析"""
    
    def __init__(self):
        self.autoencoder = None
        self.cnn_model = None
        self.lstm_model = None
    
    def build_autoencoder(self, input_dim):
        """构建自编码器"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        
        # 编码器
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # 解码器
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # 自编码器模型
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        self.autoencoder = autoencoder
        return autoencoder
    
    def detect_anomalies_with_autoencoder(self, data):
        """使用自编码器检测异常"""
        if self.autoencoder is None:
            self.build_autoencoder(data.shape[1])
        
        # 训练自编码器
        self.autoencoder.fit(data, data, epochs=100, batch_size=32, verbose=0)
        
        # 重构数据
        reconstructed = self.autoencoder.predict(data)
        
        # 计算重构误差
        reconstruction_errors = np.mean((data - reconstructed) ** 2, axis=1)
        
        # 基于误差检测异常
        threshold = np.percentile(reconstruction_errors, 95)
        anomalies = reconstruction_errors > threshold
        
        return anomalies, reconstruction_errors
```

## 📚 总结

Carranza理论为Gold-Seeker平台提供了坚实的理论基础，其主要贡献包括：

1. **系统化的方法论**: 从数据预处理到最终预测的完整流程
2. **多学科融合**: 结合地球化学、统计学、GIS和机器学习
3. **实用性导向**: 理论方法可直接应用于实际勘探工作
4. **可扩展性**: 理论框架可以容纳新的方法和技术

Gold-Seeker平台在Carranza理论基础上，进一步发展了：

1. **多智能体架构**: 实现了专业化的智能分工
2. **自动化流程**: 减少了人工干预，提高了效率
3. **集成化分析**: 支持多种方法的综合应用
4. **智能化决策**: 基于AI的智能分析和建议

这种理论与实践的结合，使得Gold-Seeker成为现代地球化学勘探的强大工具。