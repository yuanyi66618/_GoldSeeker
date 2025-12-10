"""
FractalAnomalyFilter - 分形异常滤波器

基于Carranza和Cheng的分形理论实现C-A (Concentration-Area)分形模型，
用于地球化学异常阈值确定和异常分离。

核心功能：
- plot_ca_loglog(): 绘制C-A双对数图
- calculate_threshold_interactive(): 计算分形拐点
- filter_anomalies(): 分形异常滤波
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class FractalAnomalyFilter:
    """
    分形异常滤波器
    
    基于Cheng (1999) 和 Carranza (2009) 的分形理论，
    实现C-A分形模型用于地球化学异常阈值确定。
    
    参考文献：
    1. Cheng, Q., Agterberg, F.P., Ballantyne, S.B. (1994). 
       The separation of geochemical anomalies from background by fractal methods.
    2. Carranza, E.J.M. (2009). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS.
    """
    
    def __init__(self, min_samples: int = 50):
        """
        初始化分形滤波器
        
        Args:
            min_samples: 最小样本数量要求
        """
        self.min_samples = min_samples
        self.ca_results = {}
        self.thresholds = {}
        
    def plot_ca_loglog(self, concentrations: np.ndarray,
                      areas: Optional[np.ndarray] = None,
                      element_name: str = 'Element',
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制C-A双对数图
        
        根据Cheng et al. (1994) 方法，绘制浓度-面积双对数图，
        用于识别分形拐点和确定异常阈值。
        
        Args:
            concentrations: 浓度值数组
            areas: 对应的面积数组（可选，如未提供则自动计算）
            element_name: 元素名称
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: C-A双对数图
            
        Example:
            >>> filter = FractalAnomalyFilter()
            >>> fig = filter.plot_ca_loglog(
            ...     au_concentrations, 
            ...     element_name='Au'
            ... )
            >>> fig.show()
        """
        # 数据验证
        if len(concentrations) < self.min_samples:
            raise ValueError(f"需要至少 {self.min_samples} 个样本点")
        
        # 移除缺失值和零值
        valid_mask = (concentrations > 0) & ~np.isnan(concentrations)
        conc_clean = concentrations[valid_mask]
        
        if len(conc_clean) < self.min_samples:
            raise ValueError(f"有效数据点不足 {self.min_samples} 个")
        
        # 计算面积（如果未提供）
        if areas is None:
            # 按浓度排序并计算累积面积
            sorted_conc = np.sort(conc_clean)[::-1]  # 降序排列
            cumulative_area = np.arange(1, len(sorted_conc) + 1)
            areas = cumulative_area
        
        # 创建浓度-面积数据对
        # 对浓度进行分箱以减少噪声
        n_bins = min(50, len(conc_clean) // 5)
        conc_bins = np.percentile(conc_clean, np.linspace(0, 100, n_bins))
        
        # 计算每个浓度级别的面积
        ca_data = []
        for threshold in conc_bins:
            area_above_threshold = np.sum(conc_clean >= threshold)
            if area_above_threshold > 0:
                ca_data.append([threshold, area_above_threshold])
        
        ca_data = np.array(ca_data)
        
        # 移除零值和无效值
        valid_ca = (ca_data[:, 0] > 0) & (ca_data[:, 1] > 0)
        conc_values = ca_data[valid_ca, 0]
        area_values = ca_data[valid_ca, 1]
        
        # 对数变换
        log_conc = np.log10(conc_values)
        log_area = np.log10(area_values)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'C-A Fractal Analysis for {element_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 原始C-A图
        axes[0, 0].plot(conc_values, area_values, 'bo-', alpha=0.7)
        axes[0, 0].set_xlabel('Concentration')
        axes[0, 0].set_ylabel('Area (Number of samples)')
        axes[0, 0].set_title('Concentration-Area Relationship')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 双对数图
        axes[0, 1].plot(log_conc, log_area, 'ro-', alpha=0.7)
        axes[0, 1].set_xlabel('log₁₀(Concentration)')
        axes[0, 1].set_ylabel('log₁₀(Area)')
        axes[0, 1].set_title('Log-Log C-A Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 拐点检测（初步）
        # 计算二阶导数寻找拐点
        if len(log_conc) > 10:
            # 使用滑动窗口计算局部斜率
            window_size = min(5, len(log_conc) // 4)
            slopes = []
            slope_positions = []
            
            for i in range(len(log_conc) - window_size + 1):
                x_window = log_conc[i:i+window_size]
                y_window = log_area[i:i+window_size]
                slope, _, _, _, _ = stats.linregress(x_window, y_window)
                slopes.append(slope)
                slope_positions.append(i + window_size // 2)
            
            # 绘制斜率变化
            axes[1, 0].plot(slope_positions, slopes, 'g-o', alpha=0.7)
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Local Slope')
            axes[1, 0].set_title('Slope Variation Along C-A Curve')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 标记可能的拐点
            if len(slopes) > 5:
                slope_diff = np.diff(slopes)
                inflection_candidates = np.where(np.abs(slope_diff) > np.percentile(np.abs(slope_diff), 80))[0]
                
                for candidate in inflection_candidates:
                    if candidate < len(log_conc):
                        axes[0, 1].plot(log_conc[candidate], log_area[candidate], 
                                       'r*', markersize=15, label='Inflection Point' if candidate == inflection_candidates[0] else '')
        
        # 4. 浓度分布直方图
        axes[1, 1].hist(conc_clean, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Concentration')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Concentration Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Samples: {len(conc_clean)}\n'
        stats_text += f'Mean: {np.mean(conc_clean):.3f}\n'
        stats_text += f'Std: {np.std(conc_clean):.3f}\n'
        stats_text += f'Median: {np.median(conc_clean):.3f}'
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存结果
        self.ca_results[element_name] = {
            'concentrations': conc_clean,
            'areas': area_values,
            'log_concentrations': log_conc,
            'log_areas': log_area,
            'ca_data': ca_data,
            'figure': fig
        }
        
        return fig
    
    def calculate_threshold_interactive(self, concentrations: np.ndarray,
                                      element_name: str = 'Element',
                                      method: str = 'knee',
                                      n_segments: int = 2) -> Dict[str, Any]:
        """
        计算分形拐点作为异常阈值
        
        基于Cheng et al. (1994) 的分形理论，通过识别C-A曲线的
        拐点来确定地球化学异常的阈值。
        
        Args:
            concentrations: 浓度值数组
            element_name: 元素名称
            method: 拐点检测方法 ('knee', 'kmeans', 'piecewise')
            n_segments: 分段线性拟合的段数
            
        Returns:
            Dict: 包含阈值和分析结果的字典
            
        Methods:
        - knee: 基于曲率的拐点检测
        - kmeans: K-means聚类
        - piecewise: 分段线性拟合
        
        Example:
            >>> result = filter.calculate_threshold_interactive(
            ...     au_concentrations,
            ...     element_name='Au',
            ...     method='knee'
            ... )
            >>> print(f"异常阈值: {result['threshold']:.3f}")
            >>> print(f"异常样品数: {result['n_anomalies']}")
        """
        # 获取C-A数据
        if element_name not in self.ca_results:
            self.plot_ca_loglog(concentrations, element_name=element_name)
        
        ca_data = self.ca_results[element_name]
        log_conc = ca_data['log_concentrations']
        log_area = ca_data['log_areas']
        conc_values = ca_data['concentrations']
        
        threshold = None
        method_info = {}
        
        if method == 'knee':
            # 基于曲率的拐点检测
            # 计算曲率
            if len(log_conc) > 10:
                # 平滑数据
                from scipy.signal import savgol_filter
                window_length = min(11, len(log_conc) // 3 * 2 + 1)
                if window_length % 2 == 0:
                    window_length -= 1
                
                smoothed_log_area = savgol_filter(log_area, window_length, 3)
                
                # 计算一阶和二阶导数
                first_derivative = np.gradient(smoothed_log_area, log_conc)
                second_derivative = np.gradient(first_derivative, log_conc)
                
                # 计算曲率
                curvature = np.abs(second_derivative) / (1 + first_derivative**2)**1.5
                
                # 寻找最大曲率点
                knee_idx = np.argmax(curvature)
                threshold_idx = knee_idx
                
                if threshold_idx < len(conc_values):
                    threshold = conc_values[threshold_idx]
                    method_info = {
                        'knee_index': knee_idx,
                        'curvature_max': curvature[knee_idx],
                        'first_derivative_at_knee': first_derivative[knee_idx]
                    }
        
        elif method == 'kmeans':
            # K-means聚类方法
            if len(log_conc) > 10:
                # 准备数据点
                data_points = np.column_stack([log_conc, log_area])
                
                # 使用K-means聚类
                kmeans = KMeans(n_clusters=n_segments, random_state=42)
                cluster_labels = kmeans.fit_predict(data_points)
                
                # 找到聚类之间的分界点
                cluster_boundaries = []
                for i in range(n_segments):
                    cluster_mask = cluster_labels == i
                    cluster_conc = log_conc[cluster_mask]
                    if len(cluster_conc) > 0:
                        cluster_boundaries.append(np.max(cluster_conc))
                
                cluster_boundaries.sort()
                
                # 选择最大的分界点作为阈值
                if len(cluster_boundaries) > 1:
                    threshold_idx = np.argmin(np.abs(conc_values - 10**cluster_boundaries[-2]))
                    threshold = conc_values[threshold_idx]
                    
                    method_info = {
                        'cluster_centers': kmeans.cluster_centers_,
                        'cluster_labels': cluster_labels,
                        'boundaries': cluster_boundaries
                    }
        
        elif method == 'piecewise':
            # 分段线性拟合
            if len(log_conc) > 20:
                def piecewise_linear_fit(x, y, n_segments):
                    """分段线性拟合"""
                    n_points = len(x)
                    segment_size = n_points // n_segments
                    
                    # 寻找最佳分割点
                    best_split = segment_size
                    best_error = float('inf')
                    
                    for split in range(segment_size, n_points - segment_size):
                        # 第一段
                        x1, y1 = x[:split], y[:split]
                        slope1, intercept1, _, _, _ = stats.linregress(x1, y1)
                        y1_pred = slope1 * x1 + intercept1
                        error1 = np.sum((y1 - y1_pred)**2)
                        
                        # 第二段
                        x2, y2 = x[split:], y[split:]
                        slope2, intercept2, _, _, _ = stats.linregress(x2, y2)
                        y2_pred = slope2 * x2 + intercept2
                        error2 = np.sum((y2 - y2_pred)**2)
                        
                        total_error = error1 + error2
                        
                        if total_error < best_error:
                            best_error = total_error
                            best_split = split
                    
                    return best_split, best_error
                
                # 执行分段拟合
                split_idx, split_error = piecewise_linear_fit(log_conc, log_area, n_segments)
                
                if split_idx < len(conc_values):
                    threshold = conc_values[split_idx]
                    
                    # 计算两段的斜率
                    slope1, _, _, _, _ = stats.linregress(log_conc[:split_idx], log_area[:split_idx])
                    slope2, _, _, _, _ = stats.linregress(log_conc[split_idx:], log_area[split_idx:])
                    
                    method_info = {
                        'split_index': split_idx,
                        'split_error': split_error,
                        'slope1': slope1,
                        'slope2': slope2,
                        'fractal_dimensions': [-slope1, -slope2]  # 分形维数
                    }
        
        # 如果没有找到阈值，使用统计方法
        if threshold is None:
            # 使用95百分位数作为后备阈值
            threshold = np.percentile(concentrations, 95)
            method_info = {'fallback_method': '95th_percentile'}
        
        # 计算异常统计
        anomalies = concentrations >= threshold
        n_anomalies = np.sum(anomalies)
        anomaly_percentage = n_anomalies / len(concentrations) * 100
        
        # 保存阈值
        self.thresholds[element_name] = threshold
        
        return {
            'threshold': threshold,
            'method': method,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'method_info': method_info,
            'anomaly_mask': anomalies,
            'statistics': {
                'mean': np.mean(concentrations),
                'std': np.std(concentrations),
                'median': np.median(concentrations),
                'min': np.min(concentrations),
                'max': np.max(concentrations)
            }
        }
    
    def filter_anomalies(self, concentrations: np.ndarray,
                        threshold: Optional[float] = None,
                        element_name: str = 'Element',
                        method: str = 'binary') -> Dict[str, Any]:
        """
        分形异常滤波
        
        Args:
            concentrations: 浓度值数组
            threshold: 异常阈值（如未提供则自动计算）
            element_name: 元素名称
            method: 滤波方法 ('binary', 'fuzzy', 'probability')
            
        Returns:
            Dict: 滤波结果
            
        Example:
            >>> result = filter.filter_anomalies(
            ...     au_concentrations,
            ...     threshold=2.5,
            ...     method='binary'
            ... )
            >>> print(f"异常样品: {result['anomaly_indices']}")
        """
        if threshold is None:
            if element_name in self.thresholds:
                threshold = self.thresholds[element_name]
            else:
                threshold_result = self.calculate_threshold_interactive(
                    concentrations, element_name=element_name
                )
                threshold = threshold_result['threshold']
        
        # 移除缺失值
        valid_mask = ~np.isnan(concentrations)
        conc_clean = concentrations[valid_mask]
        
        if method == 'binary':
            # 二值滤波
            anomaly_mask = conc_clean >= threshold
            anomaly_scores = anomaly_mask.astype(float)
            
        elif method == 'fuzzy':
            # 模糊隶属度函数
            # 使用S型隶属函数
            def fuzzy_membership(x, t, width=0.2):
                """S型隶属函数"""
                return 1 / (1 + np.exp(-10 * (x - t) / (t * width)))
            
            anomaly_scores = fuzzy_membership(conc_clean, threshold)
            anomaly_mask = anomaly_scores >= 0.5
            
        elif method == 'probability':
            # 基于概率的异常检测
            # 假设背景服从对数正态分布
            log_conc = np.log10(conc_clean[conc_clean > 0])
            
            if len(log_conc) > 10:
                mu, sigma = stats.norm.fit(log_conc)
                log_threshold = np.log10(threshold)
                
                # 计算p值
                p_values = 1 - stats.norm.cdf(log_conc, mu, sigma)
                anomaly_scores = p_values
                anomaly_mask = p_values < 0.05  # 5%显著性水平
            else:
                # 回退到二值方法
                anomaly_mask = conc_clean >= threshold
                anomaly_scores = anomaly_mask.astype(float)
        
        # 获取异常样品的原始索引
        anomaly_indices = np.where(valid_mask)[0][anomaly_mask]
        
        return {
            'threshold': threshold,
            'method': method,
            'anomaly_mask': anomaly_mask,
            'anomaly_scores': anomaly_scores,
            'anomaly_indices': anomaly_indices,
            'n_anomalies': len(anomaly_indices),
            'anomaly_percentage': len(anomaly_indices) / len(conc_clean) * 100,
            'anomaly_values': conc_clean[anomaly_mask],
            'background_values': conc_clean[~anomaly_mask]
        }
    
    def compare_methods(self, concentrations: np.ndarray,
                       element_name: str = 'Element',
                       methods: List[str] = ['knee', 'kmeans', 'piecewise']) -> Dict[str, Any]:
        """
        比较不同的阈值确定方法
        
        Args:
            concentrations: 浓度值数组
            element_name: 元素名称
            methods: 要比较的方法列表
            
        Returns:
            Dict: 各方法的结果比较
            
        Example:
            >>> comparison = filter.compare_methods(
            ...     au_concentrations,
            ...     element_name='Au',
            ...     methods=['knee', 'kmeans', '95th_percentile']
            ... )
            >>> for method, result in comparison['results'].items():
            ...     print(f"{method}: {result['threshold']:.3f}")
        """
        results = {}
        
        for method in methods:
            try:
                if method == '95th_percentile':
                    threshold = np.percentile(concentrations, 95)
                    filter_result = self.filter_anomalies(
                        concentrations, threshold=threshold, element_name=element_name
                    )
                    results[method] = {
                        'threshold': threshold,
                        'n_anomalies': filter_result['n_anomalies'],
                        'anomaly_percentage': filter_result['anomaly_percentage']
                    }
                else:
                    threshold_result = self.calculate_threshold_interactive(
                        concentrations, element_name=element_name, method=method
                    )
                    results[method] = threshold_result
                    
            except Exception as e:
                results[method] = {'error': str(e)}
        
        # 创建比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Threshold Method Comparison for {element_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 阈值比较
        method_names = []
        thresholds = []
        for method, result in results.items():
            if 'error' not in result:
                method_names.append(method)
                thresholds.append(result['threshold'])
        
        if thresholds:
            axes[0, 0].bar(method_names, thresholds, alpha=0.7)
            axes[0, 0].set_ylabel('Threshold Value')
            axes[0, 0].set_title('Threshold Comparison')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 异常样品数比较
        anomaly_counts = []
        for method, result in results.items():
            if 'error' not in result:
                anomaly_counts.append(result['n_anomalies'])
        
        if anomaly_counts:
            axes[0, 1].bar(method_names, anomaly_counts, alpha=0.7, color='orange')
            axes[0, 1].set_ylabel('Number of Anomalies')
            axes[0, 1].set_title('Anomaly Count Comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 浓度分布与阈值
        axes[1, 0].hist(concentrations, bins=30, alpha=0.7, edgecolor='black')
        for method, result in results.items():
            if 'error' not in result:
                axes[1, 0].axvline(x=result['threshold'], label=method, linewidth=2)
        axes[1, 0].set_xlabel('Concentration')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Concentration Distribution with Thresholds')
        axes[1, 0].legend()
        
        # 4. 方法总结表
        method_data = []
        for method, result in results.items():
            if 'error' not in result:
                method_data.append([
                    method,
                    f"{result['threshold']:.3f}",
                    result['n_anomalies'],
                    f"{result['anomaly_percentage']:.1f}%"
                ])
        
        if method_data:
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=method_data,
                                    colLabels=['Method', 'Threshold', 'Anomalies', 'Percentage'],
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        return {
            'results': results,
            'comparison_plot': fig,
            'best_method': min(results.keys(), 
                             key=lambda x: results[x]['threshold'] if 'error' not in results[x] else float('inf'))
        }