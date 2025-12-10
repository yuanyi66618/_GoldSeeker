"""
GeochemProcessor - 地球化学数据清洗与变换

基于Carranza理论实现地球化学数据的预处理，包括：
1. 检测限数据处理
2. 中心对数比变换(CLR)
3. 异常值检测与处理
4. 数据标准化

核心功能：
- impute_censored_data(): 处理低于检测限数据
- transform_clr(): 中心对数比变换
- detect_outliers(): 异常值检测
- standardize_data(): 数据标准化
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore')


class GeochemProcessor:
    """
    地球化学数据处理器
    
    基于Carranza (2009) 第2章方法，实现地球化学数据的
    专业预处理，为后续统计分析提供高质量数据。
    
    参考文献：
    Carranza, E.J.M. (2009). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS.
    """
    
    def __init__(self, detection_limits: Optional[Dict[str, float]] = None,
                 censoring_method: str = 'substitution'):
        """
        初始化数据处理器
        
        Args:
            detection_limits: 检测限字典 {元素: 检测限值}
            censoring_method: 检测限数据处理方法 ('substitution', 'ros', 'mle')
        """
        self.detection_limits = detection_limits or {}
        self.censoring_method = censoring_method
        self.scaler = None
        self.processing_log = []
        
    def impute_censored_data(self, df: pd.DataFrame,
                           elements: Optional[List[str]] = None,
                           method: Optional[str] = None) -> pd.DataFrame:
        """
        处理低于检测限数据
        
        根据Carranza (2009) 2.3节方法，处理地球化学数据中
        常见的检测限以下值（左截断数据）。
        
        Args:
            df: 原始地球化学数据
            elements: 待处理元素列表
            method: 处理方法 ('substitution', 'ros', 'mle')
            
        Returns:
            pd.DataFrame: 处理后的数据
            
        Methods:
        - substitution: 替代法（检测限/2或检测限/√2）
        - ros: Regression on Order Statistics
        - mle: Maximum Likelihood Estimation
        
        Example:
            >>> processor = GeochemProcessor(
            ...     detection_limits={'Au': 0.1, 'As': 1.0, 'Sb': 0.5}
            ... )
            >>> processed_data = processor.impute_censored_data(
            ...     raw_data, 
            ...     elements=['Au', 'As', 'Sb'],
            ...     method='substitution'
            ... )
        """
        if method is None:
            method = self.censoring_method
            
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        processed_df = df.copy()
        
        for element in elements:
            if element not in self.detection_limits:
                continue
                
            detection_limit = self.detection_limits[element]
            censored_mask = df[element] < detection_limit
            censored_count = censored_mask.sum()
            
            if censored_count == 0:
                continue
            
            # 记录处理信息
            self.processing_log.append({
                'element': element,
                'operation': 'censoring_imputation',
                'method': method,
                'censored_count': censored_count,
                'detection_limit': detection_limit
            })
            
            if method == 'substitution':
                # 替代法：使用检测限/2或检测限/√2
                if censored_count / len(df) > 0.5:  # 超过50%数据被截断
                    substitution_value = detection_limit / np.sqrt(2)
                else:
                    substitution_value = detection_limit / 2
                    
                processed_df.loc[censored_mask, element] = substitution_value
                
            elif method == 'ros':
                # ROS方法（简化实现）
                # 检测到的数据
                detected_data = df[element][~censored_mask].dropna()
                if len(detected_data) > 0:
                    # 对数变换
                    log_detected = np.log10(detected_data)
                    log_dl = np.log10(detection_limit)
                    
                    # 线性回归外推
                    rank = stats.rankdata(detected_data)
                    log_rank = np.log10(rank)
                    
                    if len(detected_data) > 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            log_rank, log_detected
                        )
                        
                        # 为截断数据生成估计值
                        censored_ranks = np.arange(1, censored_count + 1)
                        log_censored_estimates = slope * np.log10(censored_ranks) + intercept
                        censored_estimates = 10 ** log_censored_estimates
                        
                        # 确保不超过检测限
                        censored_estimates = np.minimum(censored_estimates, detection_limit * 0.99)
                        processed_df.loc[censored_mask, element] = censored_estimates
                    else:
                        # 回退到替代法
                        processed_df.loc[censored_mask, element] = detection_limit / 2
                        
            elif method == 'mle':
                # 最大似然估计（简化实现）
                detected_data = df[element][~censored_mask].dropna()
                if len(detected_data) > 5:
                    # 假设对数正态分布
                    log_detected = np.log10(detected_data)
                    mu_hat = log_detected.mean()
                    sigma_hat = log_detected.std(ddof=1)
                    
                    # 使用截断正态分布的期望值
                    from scipy.stats import truncnorm
                    a = (np.log10(detection_limit) - mu_hat) / sigma_hat
                    truncated_mean = mu_hat - sigma_hat * (
                        stats.norm.pdf(a) / (1 - stats.norm.cdf(a))
                    )
                    
                    censored_estimates = 10 ** truncated_mean
                    processed_df.loc[censored_mask, element] = censored_estimates
                else:
                    # 回退到替代法
                    processed_df.loc[censored_mask, element] = detection_limit / 2
        
        return processed_df
    
    def transform_clr(self, df: pd.DataFrame,
                     elements: Optional[List[str]] = None,
                     add_small_constant: float = 1e-6) -> pd.DataFrame:
        """
        中心对数比变换 (Centered Log-ratio Transformation)
        
        根据Aitchison (1986) 组成数据分析方法，消除地球化学
        数据的闭合效应，这是Carranza (2009) 推荐的预处理步骤。
        
        Args:
            df: 输入数据
            elements: 变换元素列表
            add_small_constant: 添加的小常数（避免log(0)）
            
        Returns:
            pd.DataFrame: CLR变换后的数据
            
        Mathematical Background:
        CLR变换公式: clr(x) = [ln(x₁/g(x)), ln(x₂/g(x)), ..., ln(x_D/g(x))]
        其中 g(x) = (x₁ × x₂ × ... × x_D)^(1/D) 是几何平均
        
        Example:
            >>> clr_data = processor.transform_clr(
            ...     geochem_df,
            ...     elements=['Au', 'As', 'Sb', 'Cu', 'Pb', 'Zn']
            ... )
            >>> print(f"变换后数据形状: {clr_data.shape}")
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        # 提取数据并添加小常数
        data = df[elements].copy()
        data = data + add_small_constant
        
        # 检查负值
        if (data < 0).any().any():
            raise ValueError("CLR变换要求数据必须为正值")
        
        # 计算几何平均
        geometric_mean = np.exp(np.log(data).mean(axis=1))
        
        # CLR变换
        clr_data = np.log(data.div(geometric_mean, axis=0))
        
        # 添加CLR前缀到列名
        clr_columns = [f'CLR_{elem}' for elem in elements]
        clr_df = pd.DataFrame(clr_data, columns=clr_columns, index=df.index)
        
        # 记录处理信息
        self.processing_log.append({
            'operation': 'clr_transformation',
            'elements': elements,
            'shape_before': data.shape,
            'shape_after': clr_df.shape
        })
        
        return clr_df
    
    def detect_outliers(self, df: pd.DataFrame,
                        elements: Optional[List[str]] = None,
                        method: str = 'robust',
                        contamination: float = 0.05) -> Dict[str, Any]:
        """
        异常值检测
        
        使用多种方法检测地球化学数据中的异常值，
        包括统计方法和基于协方差的方法。
        
        Args:
            df: 输入数据
            elements: 检测元素列表
            method: 检测方法 ('zscore', 'iqr', 'robust', 'elliptic')
            contamination: 异常值比例估计
            
        Returns:
            Dict: 包含异常值检测结果和可视化
            
        Example:
            >>> outliers = processor.detect_outliers(
            ...     geochem_df,
            ...     elements=['Au', 'As', 'Sb'],
            ...     method='robust'
            ... )
            >>> print(f"检测到 {len(outliers['outlier_indices'])} 个异常样品")
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        data = df[elements].copy()
        outlier_indices = set()
        outlier_scores = {}
        
        if method == 'zscore':
            # Z-score方法
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            outlier_mask = z_scores > 3
            outlier_indices.update(data[outlier_mask.any(axis=1)].index)
            outlier_scores['zscore'] = z_scores.max(axis=1)
            
        elif method == 'iqr':
            # 四分位距方法
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outlier_indices.update(data[outlier_mask.any(axis=1)].index)
            
        elif method == 'robust':
            # 基于鲁棒统计的方法
            median = data.median()
            mad = np.abs(data - median).median()
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > 3.5
            outlier_indices.update(data[outlier_mask.any(axis=1)].index)
            outlier_scores['robust_zscore'] = modified_z_scores.abs().max(axis=1)
            
        elif method == 'elliptic':
            # 椭圆包络方法
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(data.fillna(data.median()))
            outlier_mask = outlier_labels == -1
            outlier_indices.update(data[outlier_mask].index)
            outlier_scores['elliptic'] = detector.decision_function(data.fillna(data.median()))
        
        # 可视化异常值
        if len(elements) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Outlier Detection Results ({method.upper()} method)', 
                        fontsize=16, fontweight='bold')
            
            # 前两个元素的散点图
            elem1, elem2 = elements[0], elements[1]
            normal_data = data[~data.index.isin(outlier_indices)]
            outlier_data = data[data.index.isin(outlier_indices)]
            
            axes[0, 0].scatter(normal_data[elem1], normal_data[elem2], 
                             c='blue', label='Normal', alpha=0.6)
            axes[0, 0].scatter(outlier_data[elem1], outlier_data[elem2], 
                             c='red', label='Outliers', alpha=0.8)
            axes[0, 0].set_xlabel(elem1)
            axes[0, 0].set_ylabel(elem2)
            axes[0, 0].set_title(f'{elem1} vs {elem2}')
            axes[0, 0].legend()
            
            # 箱线图
            data_melted = data.melt(var_name='Element', value_name='Concentration')
            outlier_indicator = data_melted.index.isin(list(outlier_indices) * len(elements))
            data_melted['Type'] = ['Outlier' if i in outlier_indices else 'Normal' 
                                  for i in data_melted.index // len(elements)]
            
            sns.boxplot(data=data_melted, x='Element', y='Concentration', 
                       hue='Type', ax=axes[0, 1])
            axes[0, 1].set_title('Boxplot by Element')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 异常值分数分布
            if outlier_scores:
                score_name = list(outlier_scores.keys())[0]
                scores = outlier_scores[score_name]
                axes[1, 0].hist(scores, bins=30, alpha=0.7)
                axes[1, 0].axvline(x=np.percentile(scores, 95), color='r', 
                                 linestyle='--', label='95th percentile')
                axes[1, 0].set_xlabel(f'{score_name} Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Outlier Score Distribution')
                axes[1, 0].legend()
            
            # 异常值统计
            outlier_counts = data.index.isin(outlier_indices).groupby(data.index).sum()
            axes[1, 1].bar(range(len(outlier_counts)), outlier_counts.values)
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Number of Outlier Elements')
            axes[1, 1].set_title('Outlier Count per Sample')
            
            plt.tight_layout()
        else:
            fig = None
        
        return {
            'outlier_indices': list(outlier_indices),
            'outlier_scores': outlier_scores,
            'method': method,
            'contamination': contamination,
            'visualization': fig,
            'summary': {
                'total_samples': len(data),
                'outlier_samples': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(data) * 100
            }
        }
    
    def standardize_data(self, df: pd.DataFrame,
                        elements: Optional[List[str]] = None,
                        method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        数据标准化
        
        Args:
            df: 输入数据
            elements: 标准化元素列表
            method: 标准化方法 ('standard', 'robust', 'minmax')
            
        Returns:
            Tuple[pd.DataFrame, scaler]: 标准化后的数据和标准化器
            
        Example:
            >>> scaled_data, scaler = processor.standardize_data(
            ...     geochem_df, method='robust'
            ... )
            >>> print(f"标准化后均值: {scaled_data.mean().mean():.6f}")
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        data = df[elements].copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # 处理缺失值
        data_filled = data.fillna(data.median())
        
        # 标准化
        scaled_data = scaler.fit_transform(data_filled)
        scaled_df = pd.DataFrame(scaled_data, columns=elements, index=df.index)
        
        self.scaler = scaler
        
        # 记录处理信息
        self.processing_log.append({
            'operation': 'standardization',
            'method': method,
            'elements': elements,
            'scaler_params': scaler.get_params() if hasattr(scaler, 'get_params') else None
        })
        
        return scaled_df, scaler
    
    def get_processing_summary(self) -> pd.DataFrame:
        """
        获取数据处理摘要
        
        Returns:
            pd.DataFrame: 处理步骤摘要
        """
        if not self.processing_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.processing_log)
    
    def plot_data_distribution(self, df: pd.DataFrame,
                              elements: Optional[List[str]] = None,
                              plot_type: str = 'histogram',
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        绘制数据分布图
        
        Args:
            df: 输入数据
            elements: 绘图元素列表
            plot_type: 图表类型 ('histogram', 'boxplot', 'violin', 'qq')
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: 图形对象
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        data = df[elements].copy()
        n_elements = len(elements)
        
        # 计算子图布局
        cols = min(4, n_elements)
        rows = (n_elements + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_elements == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Data Distribution ({plot_type.title()})', 
                    fontsize=16, fontweight='bold')
        
        for i, element in enumerate(elements):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            element_data = data[element].dropna()
            
            if plot_type == 'histogram':
                ax.hist(element_data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                
            elif plot_type == 'boxplot':
                ax.boxplot(element_data)
                ax.set_ylabel('Value')
                
            elif plot_type == 'violin':
                sns.violinplot(y=element_data, ax=ax)
                
            elif plot_type == 'qq':
                stats.probplot(element_data, dist="norm", plot=ax)
                
            ax.set_title(f'{element}')
            
        # 隐藏多余的子图
        for i in range(n_elements, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig