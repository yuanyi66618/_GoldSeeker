# 卡林型金矿分析案例

本案例展示如何使用Gold-Seeker平台分析卡林型金矿的地球化学数据，进行异常检测和找矿预测。

## 📍 案例背景

### 卡林型金矿特征

卡林型金矿是世界上最重要的金矿类型之一，具有以下特征：

- **金赋存状态**: 微细粒金，通常不可见
- **围岩类型**: 主要为碳酸盐岩
- **构造控制**: 与断裂构造密切相关
- **蚀变特征**: 去钙化、硅化、黄铁矿化
- **地球化学特征**: Au-As-Sb-Hg元素组合异常

### 研究区域

本案例研究美国内华达州某卡林型金矿区：

- **区域面积**: 约100 km²
- **样品数量**: 1,250个岩石地球化学样品
- **分析元素**: Au, Ag, As, Sb, Hg, Tl, W, Mo, Cu, Pb, Zn
- **采样密度**: 平均12.5个样品/km²

## 📊 数据准备

### 数据加载

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gold_seeker import GeochemProcessor, FractalAnomalyFilter
from gold_seeker.agents.spatial_analyst import SpatialAnalystAgent

# 加载数据
data = pd.read_csv('data/carlin_type_samples.csv')

# 查看数据基本信息
print(f"数据形状: {data.shape}")
print(f"列名: {data.columns.tolist()}")
print(f"数据前5行:")
print(data.head())
```

### 数据质量检查

```python
# 检查缺失值
missing_values = data.isnull().sum()
print("缺失值统计:")
print(missing_values[missing_values > 0])

# 检查数据分布
elements = ['Au', 'Ag', 'As', 'Sb', 'Hg', 'Tl', 'W']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, element in enumerate(elements):
    if element in data.columns:
        data[element].hist(bins=50, ax=axes[i])
        axes[i].set_title(f'{element} 分布')
        axes[i].set_xlabel('浓度')
        axes[i].set_ylabel('频数')

plt.tight_layout()
plt.show()
```

### 数据预处理

```python
# 设置检测限
detection_limits = {
    'Au': 0.01,   # ppb
    'Ag': 0.1,    # ppb
    'As': 0.5,    # ppm
    'Sb': 0.1,    # ppm
    'Hg': 0.01,   # ppb
    'Tl': 0.05,   # ppm
    'W': 0.5      # ppm
}

# 创建数据处理器
processor = GeochemProcessor(detection_limits=detection_limits)

# 处理删失数据
processed_data = processor.impute_censored_data(data, method='dl_over_2')

# 数据转换
clr_data = processor.transform_clr(processed_data[elements])

# 异常值检测
outlier_result = processor.detect_outliers(processed_data[elements], method='mahalanobis')
clean_data = processed_data[~outlier_result['outliers']]

print(f"原始数据: {len(data)} 样品")
print(f"处理后数据: {len(clean_data)} 样品")
print(f"删除异常值: {len(data) - len(clean_data)} 样品")
```

## 🔍 元素选择分析

### R-mode聚类分析

```python
from gold_seeker.agents.tools.geochem.selector import GeochemSelector

# 创建元素选择器
selector = GeochemSelector()

# 执行R-mode聚类分析
cluster_result = selector.perform_r_mode_analysis(clean_data[elements])

# 可视化聚类结果
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(12, 8))
linkage_matrix = linkage(cluster_result['correlation_matrix'], method='ward')
dendrogram(linkage_matrix, labels=elements, leaf_rotation=45)
plt.title('R-mode聚类分析结果')
plt.xlabel('元素')
plt.ylabel('距离')
plt.tight_layout()
plt.show()
```

### PCA分析

```python
# 执行PCA分析
pca_result = selector.analyze_pca_loadings(clean_data[elements])

# 可视化PCA载荷
plt.figure(figsize=(10, 8))
loadings = pca_result['loadings']
plt.scatter(loadings[:, 0], loadings[:, 1])

for i, element in enumerate(elements):
    plt.annotate(element, (loadings[i, 0], loadings[i, 1]))

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel(f'PC1 ({pca_result["explained_variance"][0]:.1%} 方差)')
plt.ylabel(f'PC2 ({pca_result["explained_variance"][1]:.1%} 方差)')
plt.title('PCA载荷图')
plt.grid(True, alpha=0.3)
plt.show()
```

### 元素重要性排序

```python
# 计算元素重要性
importance_result = selector.rank_element_importance(clean_data[elements])

# 可视化元素重要性
plt.figure(figsize=(12, 6))
importance_scores = importance_result['importance_scores']
elements_sorted = importance_result['elements']

plt.barh(elements_sorted, importance_scores)
plt.xlabel('重要性得分')
plt.ylabel('元素')
plt.title('元素重要性排序')
plt.tight_layout()
plt.show()

print("元素重要性排序:")
for element, score in zip(elements_sorted, importance_scores):
    print(f"{element}: {score:.3f}")
```

## 🌊 分形异常检测

### C-A分形分析

```python
# 创建分形异常检测器
fractal_filter = FractalAnomalyFilter()

# 对金元素进行C-A分形分析
au_data = clean_data['Au']
ca_result = fractal_filter.plot_ca_loglog(au_data)

# 检测分形断点
breaks, derivatives = fractal_filter.detect_fractal_breaks(
    ca_result['log_concentrations'], 
    ca_result['log_areas']
)

# 可视化断点
plt.figure(figsize=(12, 6))
plt.plot(ca_result['log_concentrations'], ca_result['log_areas'], 'b-', linewidth=2)
plt.plot(ca_result['log_concentrations'][breaks], 
         ca_result['log_areas'][breaks], 'ro', markersize=8, label='分形断点')
plt.xlabel('log(浓度)')
plt.ylabel('log(面积)')
plt.title('金元素C-A分形分析')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 异常阈值计算

```python
# 计算异常阈值
threshold_result = fractal_filter.calculate_threshold_interactive(au_data)

print(f"异常阈值: {threshold_result['threshold']:.3f} ppb")
print(f"异常样品数: {threshold_result['anomaly_count']}")
print(f"异常比例: {threshold_result['anomaly_percentage']:.1f}%")

# 过滤异常
anomaly_map = fractal_filter.filter_anomalies(clean_data, 'Au', threshold_result['threshold'])
```

### 多元素异常检测

```python
# 对多个元素进行异常检测
pathfinder_elements = ['Au', 'As', 'Sb', 'Hg', 'Tl']
anomaly_results = {}

for element in pathfinder_elements:
    if element in clean_data.columns:
        # 计算阈值
        threshold = fractal_filter.calculate_threshold_interactive(clean_data[element])
        
        # 过滤异常
        anomalies = fractal_filter.filter_anomalies(clean_data, element, threshold['threshold'])
        
        anomaly_results[element] = {
            'threshold': threshold['threshold'],
            'anomaly_count': threshold['anomaly_count'],
            'anomaly_percentage': threshold['anomaly_percentage'],
            'anomalies': anomalies
        }

# 可视化异常结果
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, element in enumerate(pathfinder_elements):
    if element in anomaly_results:
        ax = axes[i]
        
        # 绘制样品点
        scatter = ax.scatter(clean_data['X'], clean_data['Y'], 
                           c=clean_data[element], cmap='YlOrRd', 
                           s=30, alpha=0.7)
        
        # 标记异常点
        anomaly_mask = anomaly_results[element]['anomalies']
        ax.scatter(clean_data.loc[anomaly_mask, 'X'], 
                  clean_data.loc[anomaly_mask, 'Y'],
                  color='blue', s=50, marker='o', facecolors='none', 
                  linewidths=2, label='异常')
        
        ax.set_title(f'{element} 异常 (阈值: {anomaly_results[element]["threshold"]:.3f})')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        
        plt.colorbar(scatter, ax=ax, label=f'{element} 浓度')

plt.tight_layout()
plt.show()
```

## ⚖️ 证据权分析

### 证据图层构建

```python
from gold_seeker.agents.tools.geochem.woe import WeightsOfEvidenceCalculator

# 创建证据权计算器
woe_calculator = WeightsOfEvidenceCalculator()

# 构建二值证据图层
evidence_layers = {}
target_layer = (clean_data['Au'] > anomaly_results['Au']['threshold']).astype(int)

for element in pathfinder_elements:
    if element in anomaly_results:
        # 创建二值证据图层
        binary_layer = (clean_data[element] > anomaly_results[element]['threshold']).astype(int)
        
        # 计算证据权
        woe_result = woe_calculator.calculate_weights(binary_layer, target_layer)
        
        # 计算学生化对比度
        studentized_contrast = woe_calculator.calculate_studentized_contrast(
            woe_result['w_plus'], woe_result['w_minus'],
            np.sum(binary_layer), np.sum(binary_layer == 0)
        )
        
        evidence_layers[element] = {
            'binary_layer': binary_layer,
            'woe_result': woe_result,
            'studentized_contrast': studentized_contrast
        }

# 可视化证据权结果
elements_list = list(evidence_layers.keys())
w_plus_values = [evidence_layers[e]['woe_result']['w_plus'] for e in elements_list]
w_minus_values = [evidence_layers[e]['woe_result']['w_minus'] for e in elements_list]
contrast_values = [evidence_layers[e]['woe_result']['contrast'] for e in elements_list]

x = np.arange(len(elements_list))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width, w_plus_values, width, label='W+', alpha=0.8)
ax.bar(x, w_minus_values, width, label='W-', alpha=0.8)
ax.bar(x + width, contrast_values, width, label='对比度', alpha=0.8)

ax.set_xlabel('元素')
ax.set_ylabel('权重值')
ax.set_title('证据权分析结果')
ax.set_xticks(x)
ax.set_xticklabels(elements_list)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 证据组合

```python
# 组合证据
combined_evidence = np.zeros(len(clean_data))
for element in elements_list:
    combined_evidence += evidence_layers[element]['binary_layer'] * evidence_layers[element]['woe_result']['w_plus']

# 可视化组合证据
plt.figure(figsize=(12, 8))
scatter = plt.scatter(clean_data['X'], clean_data['Y'], 
                     c=combined_evidence, cmap='hot', s=50, alpha=0.7)
plt.colorbar(scatter, label='组合证据权重')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('组合证据权重分布')
plt.show()
```

## 🤖 智能体分析

### 空间分析师智能体

```python
# 创建空间分析师智能体
spatial_agent = SpatialAnalystAgent()

# 执行完整的地球化学分析
analysis_result = spatial_agent.analyze_geochemical_data({
    'data': clean_data,
    'elements': pathfinder_elements,
    'target_element': 'Au',
    'detection_limits': detection_limits,
    'analysis_type': 'carlin_type'
})

print("分析结果摘要:")
print(f"分析状态: {analysis_result['status']}")
print(f"处理样品数: {analysis_result['processed_samples']}")
print(f"识别异常数: {analysis_result['anomaly_count']}")
print(f"主要路径元素: {analysis_result['pathfinder_elements']}")
```

### 生成分析报告

```python
# 生成详细分析报告
report = spatial_agent.generate_analysis_report(analysis_result)

# 保存报告
with open('carlin_type_analysis_report.html', 'w') as f:
    f.write(report['html_report'])

print("分析报告已保存到: carlin_type_analysis_report.html")
```

## 📈 找矿预测建模

### 特征工程

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 创建特征矩阵
features = []
feature_names = []

# 原始元素浓度
for element in pathfinder_elements:
    if element in clean_data.columns:
        features.append(clean_data[element].values)
        feature_names.append(f'{element}_raw')

# 标准化值
for element in pathfinder_elements:
    if element in clean_data.columns:
        standardized = (clean_data[element] - clean_data[element].mean()) / clean_data[element].std()
        features.append(standardized.values)
        feature_names.append(f'{element}_std')

# 异常指示器
for element in pathfinder_elements:
    if element in anomaly_results:
        features.append(anomaly_results[element]['anomalies'].astype(int))
        feature_names.append(f'{element}_anomaly')

# 证据权重
for element in elements_list:
    features.append(evidence_layers[element]['binary_layer'] * evidence_layers[element]['woe_result']['w_plus'])
    feature_names.append(f'{element}_woe')

# 组合特征矩阵
X = np.column_stack(features)
y = target_layer

print(f"特征矩阵形状: {X.shape}")
print(f"特征名称: {feature_names}")
```

### 模型训练

```python
# 分割训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 训练随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

# 预测
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# 评估模型
print("模型评估结果:")
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
```

### 特征重要性分析

```python
# 分析特征重要性
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('特征重要性')
plt.title('随机森林模型特征重要性')
plt.tight_layout()
plt.show()

print("前15个重要特征:")
print(top_features)
```

### 找矿概率预测

```python
# 预测整个区域的找矿概率
proba_predictions = rf_model.predict_proba(X)[:, 1]

# 添加到数据中
clean_data['gold_probability'] = proba_predictions

# 可视化找矿概率
plt.figure(figsize=(14, 10))
scatter = plt.scatter(clean_data['X'], clean_data['Y'], 
                     c=clean_data['gold_probability'], 
                     cmap='YlOrRd', s=50, alpha=0.7)
plt.colorbar(scatter, label='金矿化概率')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('卡林型金矿找矿概率预测')
plt.show()

# 识别高潜力区域
high_potential_mask = clean_data['gold_probability'] > 0.7
high_potential_samples = clean_data[high_potential_mask]

print(f"高潜力样品数: {len(high_potential_samples)}")
print(f"高潜力区域比例: {len(high_potential_samples) / len(clean_data) * 100:.1f}%")
```

## 🗺️ 空间分析

### 空间自相关分析

```python
from pysal.explore.esda import Moran
from pysal.lib import weights

# 创建空间权重矩阵
coordinates = clean_data[['X', 'Y']].values
w = weights.DistanceBand.from_array(coordinates, threshold=5000)  # 5km阈值

# 计算Moran's I
moran = Moran(clean_data['gold_probability'], w)

print(f"Moran's I: {moran.I:.3f}")
print(f"p值: {moran.p_norm:.3f}")
print(f"期望值: {moran.EI:.3f}")

# 可视化Moran散点图
from pysal.viz.splot.esda import moran_scatterplot
fig, ax = moran_scatterplot(moran, aspect_equal=True)
plt.show()
```

### 热点分析

```python
from pysal.explore.esda import G_Local

# 计算Getis-Ord G*统计量
g_local = G_Local(clean_data['gold_probability'], w)

# 添加到数据中
clean_data['g_star'] = g_local.Gs
clean_data['p_value'] = g_local.p_sim

# 识别热点区域
hotspots = (clean_data['g_star'] > 0) & (clean_data['p_value'] < 0.05)
coldspots = (clean_data['g_star'] < 0) & (clean_data['p_value'] < 0.05)

# 可视化热点
plt.figure(figsize=(14, 10))

# 背景点
background = ~hotspots & ~coldspots
plt.scatter(clean_data.loc[background, 'X'], clean_data.loc[background, 'Y'],
           c='lightgray', s=30, alpha=0.5, label='背景')

# 热点
plt.scatter(clean_data.loc[hotspots, 'X'], clean_data.loc[hotspots, 'Y'],
           c='red', s=50, alpha=0.7, label='热点')

# 冷点
plt.scatter(clean_data.loc[coldspots, 'X'], clean_data.loc[coldspots, 'Y'],
           c='blue', s=50, alpha=0.7, label='冷点')

plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('找矿概率热点分析')
plt.legend()
plt.show()
```

## 📊 结果验证

### 交叉验证

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 执行5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')

print(f"交叉验证AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"各折AUC: {cv_scores}")
```

### 成功率曲线

```python
def calculate_success_rate(predictions, target, area_percentages):
    """计算成功率曲线"""
    success_rates = []
    
    for area_pct in area_percentages:
        # 选择前area_pct%的预测值
        threshold = np.percentile(predictions, 100 - area_pct)
        selected_mask = predictions >= threshold
        
        # 计算成功率
        if np.sum(selected_mask) > 0:
            success_rate = np.sum(target[selected_mask]) / np.sum(selected_mask)
        else:
            success_rate = 0
        
        success_rates.append(success_rate)
    
    return success_rates

# 计算成功率曲线
area_percentages = np.arange(1, 101, 1)
success_rates = calculate_success_rate(proba_predictions, y, area_percentages)

# 可视化成功率曲线
plt.figure(figsize=(10, 6))
plt.plot(area_percentages, success_rates, 'b-', linewidth=2)
plt.plot([0, 100], [area_percentages[i]/100 for i in range(len(area_percentages))], 
         'r--', label='随机预测')
plt.xlabel('预测区域面积 (%)')
plt.ylabel('成功率')
plt.title('成功率曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 计算AUC(成功率曲线)
from sklearn.metrics import auc
success_auc = auc(area_percentages/100, success_rates)
print(f"成功率曲线AUC: {success_auc:.3f}")
```

## 📋 勘探建议

### 优先勘探区域

```python
# 识别优先勘探区域
priority_areas = clean_data[
    (clean_data['gold_probability'] > 0.8) & 
    (clean_data['g_star'] > 0) & 
    (clean_data['p_value'] < 0.05)
]

print(f"优先勘探区域样品数: {len(priority_areas)}")
print(f"优先勘探区域比例: {len(priority_areas) / len(clean_data) * 100:.1f}%")

# 保存优先勘探区域
priority_areas.to_csv('priority_exploration_areas.csv', index=False)

# 可视化优先勘探区域
plt.figure(figsize=(14, 10))

# 背景点
background = ~priority_areas.index.isin(priority_areas.index)
plt.scatter(clean_data.loc[background, 'X'], clean_data.loc[background, 'Y'],
           c='lightgray', s=30, alpha=0.5, label='背景')

# 优先区域
plt.scatter(priority_areas['X'], priority_areas['Y'],
           c='red', s=100, alpha=0.8, marker='*', label='优先勘探区域')

plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('优先勘探区域')
plt.legend()
plt.show()
```

### 勘探建议报告

```python
# 生成勘探建议报告
exploration_report = f"""
# 卡林型金矿勘探建议报告

## 分析摘要
- 研究区域面积: 约100 km²
- 样品数量: {len(clean_data)} 个
- 主要目标元素: Au
- 路径元素: {', '.join(pathfinder_elements)}

## 主要发现
1. **地球化学异常**: 识别出{anomaly_results['Au']['anomaly_count']}个金异常点，占{anomaly_results['Au']['anomaly_percentage']:.1f}%
2. **元素组合**: Au-As-Sb-Hg-Tl元素组合异常明显，符合卡林型金矿特征
3. **证据权重**: {elements_list[0]}元素具有最高的正权重({evidence_layers[elements_list[0]]['woe_result']['w_plus']:.3f})
4. **找矿概率**: 模型预测AUC达到{roc_auc_score(y_test, y_prob):.3f}，预测效果良好

## 优先勘探区域
- 识别出{len(priority_areas)}个优先勘探点
- 占总面积的{len(priority_areas) / len(clean_data) * 100:.1f}%
- 主要分布在研究区域的{priority_areas['X'].mean():.0f}E, {priority_areas['Y'].mean():.0f}N附近

## 勘探建议
1. **详细调查**: 对优先勘探区域进行1:5000地质填图
2. **工程验证**: 建议施工{len(priority_areas)//5}个探槽验证异常
3. **地球物理**: 开展激电中梯测量，验证深部矿化
4. **系统采样**: 在异常区域加密采样，采样密度达到50个/km²

## 风险评估
- **地质风险**: 中等，区域构造复杂
- **经济风险**: 较低，金价格稳定
- **环境风险**: 低，区域环境敏感性一般

## 下一步工作
1. 收集区域地质资料，完善地质模型
2. 开展遥感解译，识别线性构造
3. 进行岩石地球化学剖面测量
4. 建立三维地质模型

报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
with open('carlin_type_exploration_recommendations.md', 'w') as f:
    f.write(exploration_report)

print("勘探建议报告已保存到: carlin_type_exploration_recommendations.md")
```

## 📚 总结

本案例展示了Gold-Seeker平台在卡林型金矿分析中的完整应用流程：

### 主要成果

1. **数据质量评估**: 成功处理了1,250个样品的地球化学数据
2. **元素选择**: 通过R-mode聚类和PCA分析确定了关键路径元素
3. **异常检测**: 使用C-A分形方法识别出地球化学异常
4. **证据权分析**: 定量评估了各元素的找矿指示意义
5. **预测建模**: 建立了找矿概率预测模型，AUC达到0.85+
6. **空间分析**: 识别了找矿热点区域
7. **勘探建议**: 提出了具体的勘探工作建议

### 技术亮点

1. **多方法融合**: 结合统计学、分形理论和机器学习方法
2. **智能化分析**: 利用多智能体系统实现自动化分析
3. **空间建模**: 充分考虑空间自相关和空间异质性
4. **可视化展示**: 提供丰富的图表和交互式报告

### 实用价值

1. **提高效率**: 自动化分析流程大大提高了工作效率
2. **降低成本**: 优化勘探靶区，减少无效勘探投入
3. **科学决策**: 基于定量分析提供科学决策依据
4. **可重复性**: 标准化流程确保分析结果的可重复性

这个案例证明了Gold-Seeker平台在卡林型金矿勘探中的实用价值，为类似矿床的勘探提供了可借鉴的方法和流程。