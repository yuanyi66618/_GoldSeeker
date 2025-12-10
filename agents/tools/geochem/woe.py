"""
WeightsOfEvidenceCalculator - 证据权计算器

基于Carranza (2009) 第6章方法实现证据权法计算，
包括W+、W-、对比度C和Studentized C的计算。

核心功能：
- calculate_studentized_contrast(): 计算Studentized对比度
- calculate_weights(): 计算证据权
- validate_significance(): 统计显著性检验
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class WeightsOfEvidenceCalculator:
    """
    证据权计算器
    
    基于Carranza (2009) 第6章方法，实现证据权法(WofE)的
    完整计算流程，包括统计显著性检验。
    
    参考文献：
    Carranza, E.J.M. (2009). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS.
    """
    
    def __init__(self, min_cell_size: int = 10):
        """
        初始化证据权计算器
        
        Args:
            min_cell_size: 最小单元大小（避免零值问题）
        """
        self.min_cell_size = min_cell_size
        self.woe_results = {}
        
    def calculate_studentized_contrast(self, evidence_layer: Union[np.ndarray, pd.Series],
                                       training_points: Union[np.ndarray, pd.Series],
                                       study_area_mask: Optional[np.ndarray] = None,
                                       evidence_classes: Optional[List[float]] = None) -> pd.DataFrame:
        """
        计算Studentized对比度
        
        根据Carranza (2009) 公式6.8-6.11，计算证据权W+、W-、
        对比度C及其Studentized值，用于评估证据层的统计显著性。
        
        Args:
            evidence_layer: 证据层数据（连续值或分类值）
            training_points: 训练点（1=矿点，0=非矿点）
            study_area_mask: 研究区域掩膜
            evidence_classes: 证据层分类阈值（如未提供则自动分类）
            
        Returns:
            pd.DataFrame: 包含W+、W-、C、Studentized C等统计量的结果
            
        Mathematical Background:
        W+ = ln(P(A|D) / P(A|D̄))  # 正权重
        W- = ln(P(Ā|D) / P(Ā|D̄))  # 负权重  
        C = W+ - W-                 # 对比度
        Studentized C = C / √(Var(W+) + Var(W-))  # Studentized对比度
        
        Example:
            >>> woe = WeightsOfEvidenceCalculator()
            >>> result = woe.calculate_studentized_contrast(
            ...     fault_density_layer,
            ...     gold_deposits,
            ...     evidence_classes=[0, 0.5, 1.0, 2.0, 5.0]
            ... )
            >>> print(result[['W_plus', 'W_minus', 'Contrast', 'Studentized_C']])
        """
        # 数据预处理
        if isinstance(evidence_layer, pd.Series):
            evidence_data = evidence_layer.values
        else:
            evidence_data = evidence_layer.copy()
            
        if isinstance(training_points, pd.Series):
            training_data = training_points.values
        else:
            training_data = training_points.copy()
        
        # 确保数据维度一致
        if evidence_data.shape != training_data.shape:
            raise ValueError("证据层和训练点维度不匹配")
        
        # 应用研究区域掩膜
        if study_area_mask is not None:
            mask = study_area_mask.astype(bool)
            evidence_data = evidence_data[mask]
            training_data = training_data[mask]
        
        # 移除缺失值
        valid_mask = ~np.isnan(evidence_data) & ~np.isnan(training_data)
        evidence_data = evidence_data[valid_mask]
        training_data = training_data[valid_mask]
        
        # 确定证据层分类
        if evidence_classes is None:
            # 自动分类（基于分位数）
            n_classes = min(5, len(np.unique(evidence_data)))
            evidence_classes = np.percentile(evidence_data, 
                                            np.linspace(0, 100, n_classes + 1))[1:]
        
        # 分类证据层
        evidence_classified = np.digitize(evidence_data, evidence_classes)
        
        # 计算基本统计量
        total_cells = len(evidence_data)
        total_deposits = np.sum(training_data == 1)
        total_non_deposits = total_cells - total_deposits
        
        if total_deposits == 0:
            raise ValueError("训练点中没有矿点")
        
        results = []
        
        # 对每个证据类别计算统计量
        for class_id in range(1, len(evidence_classes) + 2):  # 包括最后一个类别
            # 当前类别的掩膜
            class_mask = (evidence_classified == class_id)
            class_cells = np.sum(class_mask)
            
            if class_cells < self.min_cell_size:
                continue
            
            # 计算条件概率
            # P(A|D) - 矿点中该类别的概率
            deposits_in_class = np.sum(training_data[class_mask] == 1)
            p_a_given_d = deposits_in_class / total_deposits if total_deposits > 0 else 0
            
            # P(A|D̄) - 非矿点中该类别的概率
            non_deposits_in_class = class_cells - deposits_in_class
            p_a_given_not_d = non_deposits_in_class / total_non_deposits if total_non_deposits > 0 else 0
            
            # P(Ā|D) - 矿点中不在该类别的概率
            p_not_a_given_d = 1 - p_a_given_d
            
            # P(Ā|D̄) - 非矿点中不在该类别的概率
            p_not_a_given_not_d = 1 - p_a_given_not_d
            
            # 计算权重（添加小常数避免log(0)）
            epsilon = 1e-10
            
            if p_a_given_d > epsilon and p_a_given_not_d > epsilon:
                w_plus = np.log(p_a_given_d / p_a_given_not_d)
            else:
                w_plus = 0
            
            if p_not_a_given_d > epsilon and p_not_a_given_not_d > epsilon:
                w_minus = np.log(p_not_a_given_d / p_not_a_given_not_d)
            else:
                w_minus = 0
            
            # 计算对比度
            contrast = w_plus - w_minus
            
            # 计算方差（Carranza 2009, 公式6.12-6.13）
            var_w_plus = (1 / deposits_in_class if deposits_in_class > 0 else 0) + \
                        (1 / non_deposits_in_class if non_deposits_in_class > 0 else 0)
            
            var_w_minus = (1 / (total_deposits - deposits_in_class) if total_deposits - deposits_in_class > 0 else 0) + \
                          (1 / (total_non_deposits - non_deposits_in_class) if total_non_deposits - non_deposits_in_class > 0 else 0)
            
            # 计算Studentized对比度
            var_contrast = var_w_plus + var_w_minus
            studentized_contrast = contrast / np.sqrt(var_contrast) if var_contrast > 0 else 0
            
            # 计算置信度
            confidence = 2 * (1 - stats.norm.cdf(abs(studentized_contrast)))  # 双尾检验
            
            # 确定类别范围
            if class_id == 1:
                class_range = f"< {evidence_classes[0]:.3f}"
            elif class_id == len(evidence_classes) + 1:
                class_range = f">= {evidence_classes[-1]:.3f}"
            else:
                class_range = f"{evidence_classes[class_id-2]:.3f} - {evidence_classes[class_id-1]:.3f}"
            
            results.append({
                'Class_ID': class_id,
                'Class_Range': class_range,
                'Class_Cells': class_cells,
                'Deposits_in_Class': deposits_in_class,
                'Non_Deposits_in_Class': non_deposits_in_class,
                'P_A_given_D': p_a_given_d,
                'P_A_given_not_D': p_a_given_not_d,
                'W_plus': w_plus,
                'W_minus': w_minus,
                'Contrast': contrast,
                'Var_W_plus': var_w_plus,
                'Var_W_minus': var_w_minus,
                'Studentized_C': studentized_contrast,
                'Confidence': confidence,
                'Significant': abs(studentized_contrast) > 1.96  # 95%置信水平
            })
        
        result_df = pd.DataFrame(results)
        
        # 保存结果
        layer_name = getattr(evidence_layer, 'name', 'Evidence_Layer')
        self.woe_results[layer_name] = result_df
        
        return result_df
    
    def calculate_weights(self, evidence_layers: Dict[str, Union[np.ndarray, pd.Series]],
                         training_points: Union[np.ndarray, pd.Series],
                         study_area_mask: Optional[np.ndarray] = None,
                         evidence_classes: Optional[Dict[str, List[float]]] = None) -> Dict[str, pd.DataFrame]:
        """
        批量计算多个证据层的权重
        
        Args:
            evidence_layers: 证据层字典 {层名: 数据}
            training_points: 训练点
            study_area_mask: 研究区域掩膜
            evidence_classes: 各层的分类阈值
            
        Returns:
            Dict[str, pd.DataFrame]: 各层的证据权结果
            
        Example:
            >>> evidence_layers = {
            ...     'fault_density': fault_layer,
            ...     'rock_type': rock_layer,
            ...     'geochem_Au': au_layer
            ... }
            >>> results = woe.calculate_weights(
            ...     evidence_layers, gold_deposits
            ... )
            >>> for layer_name, result in results.items():
            ...     print(f"{layer_name}: {len(result)} 个类别")
        """
        all_results = {}
        
        for layer_name, layer_data in evidence_layers.items():
            try:
                # 获取该层的分类阈值
                layer_classes = None
                if evidence_classes and layer_name in evidence_classes:
                    layer_classes = evidence_classes[layer_name]
                
                # 计算证据权
                result = self.calculate_studentized_contrast(
                    layer_data, training_points, study_area_mask, layer_classes
                )
                
                # 添加层名信息
                result['Layer_Name'] = layer_name
                all_results[layer_name] = result
                
            except Exception as e:
                print(f"计算层 {layer_name} 时出错: {str(e)}")
                continue
        
        return all_results
    
    def validate_significance(self, woe_results: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                             significance_level: float = 0.05) -> Dict[str, Any]:
        """
        统计显著性检验
        
        Args:
            woe_results: 证据权计算结果
            significance_level: 显著性水平
            
        Returns:
            Dict: 显著性检验结果
            
        Example:
            >>> validation = woe.validate_significance(
            ...     woe_results, significance_level=0.05
            ... )
            >>> print(f"显著类别数: {validation['significant_classes']}")
        """
        if isinstance(woe_results, pd.DataFrame):
            woe_results = {'single_layer': woe_results}
        
        validation_results = {}
        total_classes = 0
        significant_classes = 0
        
        for layer_name, result_df in woe_results.items():
            layer_validation = {
                'total_classes': len(result_df),
                'significant_classes': np.sum(result_df['Significant']),
                'mean_studentized_c': np.mean(np.abs(result_df['Studentized_C'])),
                'max_contrast': np.max(np.abs(result_df['Contrast'])),
                'positive_contrast_classes': np.sum(result_df['Contrast'] > 0),
                'negative_contrast_classes': np.sum(result_df['Contrast'] < 0)
            }
            
            # 计算p值统计
            p_values = result_df['Confidence']
            layer_validation.update({
                'mean_p_value': np.mean(p_values),
                'min_p_value': np.min(p_values),
                'classes_below_significance': np.sum(p_values < significance_level)
            })
            
            validation_results[layer_name] = layer_validation
            total_classes += layer_validation['total_classes']
            significant_classes += layer_validation['significant_classes']
        
        # 总体统计
        validation_results['overall'] = {
            'total_layers': len(woe_results),
            'total_classes': total_classes,
            'significant_classes': significant_classes,
            'overall_significance_rate': significant_classes / total_classes if total_classes > 0 else 0
        }
        
        return validation_results
    
    def plot_woe_results(self, woe_results: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                        plot_type: str = 'weights',
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        绘制证据权结果图
        
        Args:
            woe_results: 证据权计算结果
            plot_type: 图表类型 ('weights', 'contrast', 'significance')
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: 图形对象
            
        Example:
            >>> fig = woe.plot_woe_results(
            ...     woe_results, plot_type='contrast'
            ... )
            >>> fig.show()
        """
        if isinstance(woe_results, pd.DataFrame):
            woe_results = {'single_layer': woe_results}
        
        n_layers = len(woe_results)
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Weights of Evidence Analysis ({plot_type.title()})', 
                    fontsize=16, fontweight='bold')
        
        for i, (layer_name, result_df) in enumerate(woe_results.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            if plot_type == 'weights':
                # 绘制W+和W-
                x_pos = np.arange(len(result_df))
                width = 0.35
                
                ax.bar(x_pos - width/2, result_df['W_plus'], width, 
                      label='W+', alpha=0.7, color='green')
                ax.bar(x_pos + width/2, result_df['W_minus'], width, 
                      label='W-', alpha=0.7, color='red')
                
                ax.set_xlabel('Class')
                ax.set_ylabel('Weight')
                ax.set_title(f'{layer_name}: W+ and W-')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(result_df['Class_Range'], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif plot_type == 'contrast':
                # 绘制对比度
                colors = ['red' if c < 0 else 'green' for c in result_df['Contrast']]
                bars = ax.bar(range(len(result_df)), result_df['Contrast'], 
                             color=colors, alpha=0.7)
                
                # 标记显著的类别
                for i, (idx, row) in enumerate(result_df.iterrows()):
                    if row['Significant']:
                        ax.text(i, row['Contrast'], '*', ha='center', va='bottom' if row['Contrast'] > 0 else 'top', 
                               fontsize=12, fontweight='bold')
                
                ax.set_xlabel('Class')
                ax.set_ylabel('Contrast (C)')
                ax.set_title(f'{layer_name}: Contrast Values')
                ax.set_xticks(range(len(result_df)))
                ax.set_xticklabels(result_df['Class_Range'], rotation=45, ha='right')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                
            elif plot_type == 'significance':
                # 绘制Studentized对比度
                colors = ['red' if abs(c) < 1.96 else 'green' for c in result_df['Studentized_C']]
                bars = ax.bar(range(len(result_df)), result_df['Studentized_C'], 
                             color=colors, alpha=0.7)
                
                # 添加显著性阈值线
                ax.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='95% threshold')
                ax.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                ax.set_xlabel('Class')
                ax.set_ylabel('Studentized Contrast')
                ax.set_title(f'{layer_name}: Statistical Significance')
                ax.set_xticks(range(len(result_df)))
                ax.set_xticklabels(result_df['Class_Range'], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_layers, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_response_surface(self, evidence_layers: Dict[str, Union[np.ndarray, pd.Series]],
                               woe_results: Dict[str, pd.DataFrame],
                               study_area_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建响应面（预测图）
        
        Args:
            evidence_layers: 证据层数据
            woe_results: 证据权计算结果
            study_area_mask: 研究区域掩膜
            
        Returns:
            np.ndarray: 响应面数组
            
        Example:
            >>> response_surface = woe.create_response_surface(
            ...     evidence_layers, woe_results
            ... )
            >>> print(f"响应面范围: {response_surface.min():.3f} - {response_surface.max():.3f}")
        """
        # 获取第一个证据层的形状作为参考
        first_layer = list(evidence_layers.values())[0]
        if isinstance(first_layer, pd.Series):
            response_surface = np.zeros(first_layer.shape)
        else:
            response_surface = np.zeros(first_layer.shape)
        
        # 应用研究区域掩膜
        if study_area_mask is not None:
            mask = study_area_mask.astype(bool)
        else:
            mask = np.ones_like(response_surface, dtype=bool)
        
        # 对每个证据层贡献权重
        for layer_name, layer_data in evidence_layers.items():
            if layer_name not in woe_results:
                continue
            
            # 获取该层的权重结果
            layer_woe = woe_results[layer_name]
            
            # 选择最佳类别（最大Studentized对比度的绝对值）
            best_class_idx = np.argmax(np.abs(layer_woe['Studentized_C']))
            best_class = layer_woe.iloc[best_class_idx]
            
            # 获取该类别的权重
            weight = best_class['Contrast']
            
            # 应用到响应面
            if isinstance(layer_data, pd.Series):
                layer_array = layer_data.values
            else:
                layer_array = layer_data.copy()
            
            # 只在显著类别应用权重
            if best_class['Significant']:
                # 创建该类别的掩膜
                class_thresholds = []
                for i, row in layer_woe.iterrows():
                    if i == 0:
                        class_thresholds.append((-np.inf, row['Class_Range'].split()[1]))
                    elif i == len(layer_woe) - 1:
                        class_thresholds.append((float(row['Class_Range'].split()[2]), np.inf))
                    else:
                        parts = row['Class_Range'].split()
                        class_thresholds.append((float(parts[0]), float(parts[2])))
                
                # 应用最佳类别的权重
                if best_class_idx < len(class_thresholds):
                    lower, upper = class_thresholds[best_class_idx]
                    class_mask = (layer_array >= lower) & (layer_array < upper)
                    response_surface[class_mask & mask] += weight
        
        return response_surface
    
    def get_summary_statistics(self, woe_results: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        获取汇总统计
        
        Args:
            woe_results: 证据权计算结果
            
        Returns:
            pd.DataFrame: 汇总统计表
            
        Example:
            >>> summary = woe.get_summary_statistics(woe_results)
            >>> print(summary[['Layer_Name', 'Max_Contrast', 'Significant_Classes']])
        """
        if isinstance(woe_results, pd.DataFrame):
            woe_results = {'single_layer': woe_results}
        
        summary_data = []
        
        for layer_name, result_df in woe_results.items():
            summary_data.append({
                'Layer_Name': layer_name,
                'Total_Classes': len(result_df),
                'Significant_Classes': np.sum(result_df['Significant']),
                'Max_Contrast': np.max(np.abs(result_df['Contrast'])),
                'Mean_Studentized_C': np.mean(np.abs(result_df['Studentized_C'])),
                'Positive_Contrast_Classes': np.sum(result_df['Contrast'] > 0),
                'Negative_Contrast_Classes': np.sum(result_df['Contrast'] < 0),
                'Best_Studentized_C': np.max(np.abs(result_df['Studentized_C']))
            })
        
        return pd.DataFrame(summary_data)