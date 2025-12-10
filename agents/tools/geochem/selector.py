"""
GeochemSelector - 地球化学自动特征筛选

基于Carranza的《Geochemical Anomaly and Mineral Prospectivity Mapping in GIS》
实现R型聚类分析和主成分分析，自动识别金矿指示元素组合。

核心功能：
1. R型聚类分析：识别元素共生组合
2. 主成分分析：提取找矿元素组合
3. 元素重要性排序：推荐最佳找矿指标
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class GeochemSelector:
    """
    地球化学特征选择器
    
    基于统计分析和机器学习方法，自动识别与金矿化相关的
    指示元素组合，为后续成矿预测提供最优特征集。
    
    参考文献：
    Carranza, E.J.M. (2009). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS.
    """
    
    def __init__(self, detection_limits: Optional[Dict[str, float]] = None):
        """
        初始化特征选择器
        
        Args:
            detection_limits: 元素检测限字典 {元素名: 检测限值}
        """
        self.detection_limits = detection_limits or {}
        self.correlation_matrix = None
        self.linkage_matrix = None
        self.pca_model = None
        self.element_importance = {}
        
    def perform_r_mode_analysis(self, df: pd.DataFrame, 
                              elements: Optional[List[str]] = None,
                              method: str = 'ward',
                              metric: str = 'correlation',
                              figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Any]:
        """
        执行R型聚类分析
        
        根据Carranza (2009) 第3章方法，通过元素间的相关性识别
        地球化学共生组合，揭示成矿地球化学特征。
        
        Args:
            df: 地球化学数据DataFrame (行为样品，列为元素)
            elements: 待分析元素列表，None表示使用所有数值列
            method: 聚类方法 ('ward', 'complete', 'average', 'single')
            metric: 距离度量 ('correlation', 'euclidean', 'cityblock')
            figsize: 图形尺寸
            
        Returns:
            Dict: 包含聚类结果和可视化对象的字典
            
        Example:
            >>> selector = GeochemSelector()
            >>> result = selector.perform_r_mode_analysis(
            ...     geochem_df, 
            ...     elements=['Au', 'Ag', 'As', 'Sb', 'Cu', 'Pb', 'Zn'],
            ...     method='ward'
            ... )
            >>> print(f"识别出 {len(result['clusters'])} 个元素组合")
            >>> result['dendrogram'].show()
        """
        # 数据预处理
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        # 提取数据并处理缺失值
        data = df[elements].copy()
        
        # 处理低于检测限的数据
        for element in elements:
            if element in self.detection_limits:
                data[element] = data[element].fillna(self.detection_limits[element] / 2)
            else:
                data[element] = data[element].fillna(data[element].median())
        
        # 计算相关性矩阵
        self.correlation_matrix = data.corr(method='pearson')
        
        # 转换为距离矩阵（1 - |相关系数|）
        distance_matrix = 1 - np.abs(self.correlation_matrix)
        
        # 层次聚类
        self.linkage_matrix = linkage(distance_matrix.values, method=method, metric=metric)
        
        # 绘制树状图
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(
            self.linkage_matrix,
            labels=elements,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True,
            ax=ax
        )
        
        ax.set_title('R-mode Cluster Analysis of Geochemical Elements', fontsize=14, fontweight='bold')
        ax.set_xlabel('Elements', fontsize=12)
        ax.set_ylabel('Distance (1 - |correlation|)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 确定聚类数量（基于距离阈值）
        max_distance = 0.5  # 可调整的阈值
        clusters = fcluster(self.linkage_matrix, t=max_distance, criterion='distance')
        
        # 组织聚类结果
        cluster_results = {}
        for i, (element, cluster_id) in enumerate(zip(elements, clusters)):
            if cluster_id not in cluster_results:
                cluster_results[cluster_id] = []
            cluster_results[cluster_id].append(element)
        
        # 分析每个聚类特征
        cluster_analysis = {}
        for cluster_id, cluster_elements in cluster_results.items():
            cluster_data = data[cluster_elements]
            cluster_corr = cluster_data.corr()
            
            # 计算聚类内平均相关性
            avg_correlation = cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].mean()
            
            cluster_analysis[cluster_id] = {
                'elements': cluster_elements,
                'size': len(cluster_elements),
                'avg_correlation': avg_correlation,
                'correlation_matrix': cluster_corr
            }
        
        return {
            'linkage_matrix': self.linkage_matrix,
            'correlation_matrix': self.correlation_matrix,
            'clusters': cluster_results,
            'cluster_analysis': cluster_analysis,
            'dendrogram': fig,
            'elements': elements
        }
    
    def analyze_pca_loadings(self, df: pd.DataFrame,
                            elements: Optional[List[str]] = None,
                            n_components: int = 3,
                            var_threshold: float = 0.8,
                            figsize: Tuple[int, int] = (15, 10)) -> Dict[str, Any]:
        """
        分析主成分载荷
        
        基于Carranza (2009) 第4章方法，通过主成分分析识别
        控制地球化学数据变异的主要因素，推荐最佳找矿元素组合。
        
        Args:
            df: 地球化学数据DataFrame
            elements: 待分析元素列表
            n_components: 主成分数量
            var_threshold: 方差解释阈值
            figsize: 图形尺寸
            
        Returns:
            Dict: 包含PCA结果和可视化对象的字典
            
        Example:
            >>> result = selector.analyze_pca_loadings(
            ...     geochem_df,
            ...     elements=['Au', 'As', 'Sb', 'Hg', 'Cu', 'Pb', 'Zn'],
            ...     n_components=3
            ... )
            >>> print(f"前3个主成分解释方差: {result['explained_variance_ratio']}")
            >>> result['loadings_plot'].show()
        """
        # 数据预处理
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        # 提取数据并处理缺失值
        data = df[elements].copy()
        
        # 处理低于检测限的数据
        for element in elements:
            if element in self.detection_limits:
                data[element] = data[element].fillna(self.detection_limits[element] / 2)
            else:
                data[element] = data[element].fillna(data[element].median())
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 执行PCA
        self.pca_model = PCA(n_components=n_components)
        pca_result = self.pca_model.fit_transform(data_scaled)
        
        # 获取载荷矩阵
        loadings = self.pca_model.components_.T * np.sqrt(self.pca_model.explained_variance_)
        
        # 创建载荷DataFrame
        loadings_df = pd.DataFrame(
            loadings,
            index=elements,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # 计算累积方差解释比例
        cum_var_ratio = np.cumsum(self.pca_model.explained_variance_ratio_)
        
        # 确定达到阈值所需的主成分数
        n_components_threshold = np.argmax(cum_var_ratio >= var_threshold) + 1
        
        # 可视化载荷
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('PCA Loadings Analysis for Geochemical Elements', fontsize=16, fontweight='bold')
        
        # 载荷热图
        sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
                   ax=axes[0, 0], fmt='.2f')
        axes[0, 0].set_title('Component Loadings')
        
        # 方差解释比例
        axes[0, 1].bar(range(1, n_components+1), self.pca_model.explained_variance_ratio_)
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Explained Variance Ratio')
        axes[0, 1].set_title('Individual Variance Explained')
        
        # 累积方差解释比例
        axes[1, 0].plot(range(1, n_components+1), cum_var_ratio, 'bo-')
        axes[1, 0].axhline(y=var_threshold, color='r', linestyle='--', 
                          label=f'{var_threshold*100}% Threshold')
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Cumulative Variance Explained')
        axes[1, 0].set_title('Cumulative Variance Explained')
        axes[1, 0].legend()
        
        # PC1 vs PC2载荷散点图
        if n_components >= 2:
            axes[1, 1].scatter(loadings[:, 0], loadings[:, 1])
            for i, element in enumerate(elements):
                axes[1, 1].annotate(element, (loadings[i, 0], loadings[i, 1]))
            axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 1].set_xlabel('PC1 Loading')
            axes[1, 1].set_ylabel('PC2 Loading')
            axes[1, 1].set_title('PC1 vs PC2 Loadings')
        
        plt.tight_layout()
        
        # 分析每个主成分的元素组合
        component_analysis = {}
        for i in range(n_components):
            component_loadings = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
            
            # 识别高载荷元素（|载荷| > 0.5）
            high_loading_elements = component_loadings[component_loadings > 0.5].index.tolist()
            
            component_analysis[f'PC{i+1}'] = {
                'explained_variance': self.pca_model.explained_variance_ratio_[i],
                'high_loading_elements': high_loading_elements,
                'all_loadings': component_loadings.to_dict()
            }
        
        # 推荐找矿元素组合
        # 基于PC1（通常解释最大方差）的高载荷元素
        pc1_high_loadings = component_analysis['PC1']['high_loading_elements']
        
        # 检查是否包含典型金矿指示元素
        pathfinder_elements = ['Au', 'Ag', 'As', 'Sb', 'Hg', 'Cu', 'Pb', 'Zn', 'Bi', 'Te']
        recommended_elements = [elem for elem in pc1_high_loadings if elem in pathfinder_elements]
        
        if not recommended_elements:
            # 如果没有典型指示元素，选择载荷最高的元素
            recommended_elements = pc1_high_loadings[:3]
        
        return {
            'pca_model': self.pca_model,
            'loadings': loadings_df,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance_ratio': cum_var_ratio,
            'n_components_threshold': n_components_threshold,
            'component_analysis': component_analysis,
            'recommended_elements': recommended_elements,
            'loadings_plot': fig,
            'pca_result': pca_result
        }
    
    def rank_element_importance(self, df: pd.DataFrame,
                               target_element: str = 'Au',
                               method: str = 'correlation') -> Dict[str, float]:
        """
        元素重要性排序
        
        Args:
            df: 地球化学数据
            target_element: 目标元素（通常是金）
            method: 评估方法 ('correlation', 'mutual_info', 'random_forest')
            
        Returns:
            Dict: 元素重要性字典
            
        Example:
            >>> importance = selector.rank_element_importance(
            ...     geochem_df, target_element='Au', method='correlation'
            ... )
            >>> for element, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            ...     print(f"{element}: {score:.3f}")
        """
        elements = [col for col in df.columns if col != target_element and 
                   df[col].dtype in ['float64', 'int64']]
        
        if target_element not in df.columns:
            raise ValueError(f"Target element '{target_element}' not found in data")
        
        # 处理缺失值
        data = df[[target_element] + elements].copy()
        data = data.fillna(data.median())
        
        importance_scores = {}
        
        if method == 'correlation':
            # 基于与目标元素的相关性
            for element in elements:
                corr, _ = pearsonr(data[element], data[target_element])
                importance_scores[element] = abs(corr)
        
        elif method == 'mutual_info':
            # 基于互信息
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(data[elements], data[target_element])
            importance_scores = dict(zip(elements, mi_scores))
        
        elif method == 'random_forest':
            # 基于随机森林特征重要性
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(data[elements], data[target_element])
            importance_scores = dict(zip(elements, rf.feature_importances_))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.element_importance = importance_scores
        return importance_scores
    
    def get_optimal_element_combination(self, df: pd.DataFrame,
                                      target_element: str = 'Au',
                                      max_elements: int = 5,
                                      method: str = 'correlation') -> List[str]:
        """
        获取最优元素组合
        
        Args:
            df: 地球化学数据
            target_element: 目标元素
            max_elements: 最大元素数量
            method: 重要性评估方法
            
        Returns:
            List[str]: 推荐的元素组合
            
        Example:
            >>> optimal_combo = selector.get_optimal_element_combination(
            ...     geochem_df, target_element='Au', max_elements=5
            ... )
            >>> print(f"推荐元素组合: {', '.join(optimal_combo)}")
        """
        importance = self.rank_element_importance(df, target_element, method)
        
        # 按重要性排序并选择前N个元素
        sorted_elements = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        optimal_elements = [elem for elem, score in sorted_elements[:max_elements]]
        
        return optimal_elements