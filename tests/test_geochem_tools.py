"""
地球化学处理工具测试

测试GeochemSelector、GeochemProcessor、FractalAnomalyFilter、WeightsOfEvidenceCalculator
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.tools.geochem import (
    GeochemSelector,
    GeochemProcessor,
    FractalAnomalyFilter,
    WeightsOfEvidenceCalculator
)


class TestGeochemSelector:
    """测试GeochemSelector类"""
    
    @pytest.fixture
    def selector(self):
        """创建测试用的选择器实例"""
        detection_limits = {'Au': 0.1, 'As': 1.0, 'Sb': 0.5, 'Cu': 2.0}
        return GeochemSelector(detection_limits)
    
    @pytest.fixture
    def sample_data(self):
        """创建测试用的地球化学数据"""
        np.random.seed(42)
        n_samples = 100
        
        # 模拟相关的地球化学数据
        data = {
            'Au': np.random.lognormal(0, 1, n_samples),
            'As': np.random.lognormal(1, 0.8, n_samples),
            'Sb': np.random.lognormal(0.5, 0.9, n_samples),
            'Cu': np.random.lognormal(2, 0.7, n_samples),
            'Pb': np.random.lognormal(1.5, 0.6, n_samples),
            'Zn': np.random.lognormal(1.8, 0.5, n_samples),
            'Ag': np.random.lognormal(-0.5, 1.1, n_samples),
            'Hg': np.random.lognormal(-1, 1.2, n_samples)
        }
        
        # 添加一些相关性
        data['Au'] = 0.7 * data['Au'] + 0.3 * data['As'] + np.random.normal(0, 0.1, n_samples)
        data['Sb'] = 0.6 * data['Sb'] + 0.4 * data['As'] + np.random.normal(0, 0.1, n_samples)
        
        return pd.DataFrame(data)
    
    def test_initialization(self, selector):
        """测试初始化"""
        assert selector.detection_limits == {'Au': 0.1, 'As': 1.0, 'Sb': 0.5, 'Cu': 2.0}
        assert selector.correlation_matrix is None
        assert selector.linkage_matrix is None
    
    def test_perform_r_mode_analysis(self, selector, sample_data):
        """测试R型聚类分析"""
        elements = ['Au', 'As', 'Sb', 'Cu', 'Pb', 'Zn']
        
        result = selector.perform_r_mode_analysis(
            sample_data, 
            elements=elements,
            method='ward'
        )
        
        # 验证结果结构
        assert 'linkage_matrix' in result
        assert 'correlation_matrix' in result
        assert 'clusters' in result
        assert 'cluster_analysis' in result
        assert 'dendrogram' in result
        assert 'elements' in result
        
        # 验证数据
        assert result['elements'] == elements
        assert len(result['clusters']) > 0
        assert isinstance(result['correlation_matrix'], pd.DataFrame)
        assert result['correlation_matrix'].shape == (len(elements), len(elements))
    
    def test_analyze_pca_loadings(self, selector, sample_data):
        """测试主成分分析"""
        elements = ['Au', 'As', 'Sb', 'Cu', 'Pb', 'Zn']
        
        result = selector.analyze_pca_loadings(
            sample_data,
            elements=elements,
            n_components=3
        )
        
        # 验证结果结构
        assert 'pca_model' in result
        assert 'loadings' in result
        assert 'explained_variance_ratio' in result
        assert 'cumulative_variance_ratio' in result
        assert 'component_analysis' in result
        assert 'recommended_elements' in result
        assert 'loadings_plot' in result
        
        # 验证数据
        assert len(result['explained_variance_ratio']) == 3
        assert isinstance(result['loadings'], pd.DataFrame)
        assert result['loadings'].shape == (len(elements), 3)
        assert len(result['recommended_elements']) > 0
    
    def test_rank_element_importance(self, selector, sample_data):
        """测试元素重要性排序"""
        importance = selector.rank_element_importance(
            sample_data, 
            target_element='Au',
            method='correlation'
        )
        
        # 验证结果
        assert isinstance(importance, dict)
        assert 'Au' not in importance  # 目标元素不应在结果中
        assert len(importance) == len(sample_data.columns) - 1
        
        # 验证分数范围
        for score in importance.values():
            assert 0 <= score <= 1
    
    def test_get_optimal_element_combination(self, selector, sample_data):
        """测试获取最优元素组合"""
        optimal = selector.get_optimal_element_combination(
            sample_data,
            target_element='Au',
            max_elements=4,
            method='correlation'
        )
        
        # 验证结果
        assert isinstance(optimal, list)
        assert len(optimal) <= 4
        assert 'Au' not in optimal  # 目标元素不应在组合中


class TestGeochemProcessor:
    """测试GeochemProcessor类"""
    
    @pytest.fixture
    def processor(self):
        """创建测试用的处理器实例"""
        detection_limits = {'Au': 0.1, 'As': 1.0, 'Sb': 0.5}
        return GeochemProcessor(detection_limits, censoring_method='substitution')
    
    @pytest.fixture
    def sample_data_with_censoring(self):
        """创建包含检测限以下数据的测试数据"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Au': np.concatenate([
                np.random.uniform(0, 0.05, 30),  # 低于检测限
                np.random.lognormal(0, 1, 70)    # 高于检测限
            ]),
            'As': np.concatenate([
                np.random.uniform(0, 0.8, 20),   # 低于检测限
                np.random.lognormal(1, 0.8, 80)
            ]),
            'Sb': np.random.lognormal(0.5, 0.9, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self, processor):
        """测试初始化"""
        assert processor.detection_limits == {'Au': 0.1, 'As': 1.0, 'Sb': 0.5}
        assert processor.censoring_method == 'substitution'
        assert processor.processing_log == []
    
    def test_impute_censored_data(self, processor, sample_data_with_censoring):
        """测试检测限数据处理"""
        elements = ['Au', 'As', 'Sb']
        
        result = processor.impute_censored_data(
            sample_data_with_censoring,
            elements=elements,
            method='substitution'
        )
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data_with_censoring.shape
        assert not result.isnull().any().any()  # 不应有缺失值
        
        # 验证检测限处理
        assert (result['Au'] >= 0.05).all()  # 替代值应 >= 检测限/2
        assert (result['As'] >= 0.5).all()   # 替代值应 >= 检测限/2
        
        # 验证处理日志
        assert len(processor.processing_log) > 0
        assert any(log['operation'] == 'censoring_imputation' for log in processor.processing_log)
    
    def test_transform_clr(self, processor, sample_data_with_censoring):
        """测试CLR变换"""
        elements = ['Au', 'As', 'Sb']
        
        # 先处理检测限
        processed_data = processor.impute_censored_data(sample_data_with_censoring, elements)
        
        # 进行CLR变换
        clr_result = processor.transform_clr(processed_data, elements=elements)
        
        # 验证结果
        assert isinstance(clr_result, pd.DataFrame)
        assert clr_result.shape[0] == processed_data.shape[0]
        assert clr_result.shape[1] == len(elements)
        
        # 验证CLR属性：每行和应接近0
        row_sums = clr_result.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)
        
        # 验证列名
        expected_columns = [f'CLR_{elem}' for elem in elements]
        assert list(clr_result.columns) == expected_columns
    
    def test_detect_outliers(self, processor, sample_data_with_censoring):
        """测试异常值检测"""
        elements = ['Au', 'As', 'Sb']
        
        result = processor.detect_outliers(
            sample_data_with_censoring,
            elements=elements,
            method='robust'
        )
        
        # 验证结果结构
        assert 'outlier_indices' in result
        assert 'outlier_scores' in result
        assert 'method' in result
        assert 'summary' in result
        
        # 验证数据
        assert isinstance(result['outlier_indices'], list)
        assert result['method'] == 'robust'
        assert result['summary']['total_samples'] == len(sample_data_with_censoring)
    
    def test_standardize_data(self, processor, sample_data_with_censoring):
        """测试数据标准化"""
        elements = ['Au', 'As', 'Sb']
        
        scaled_data, scaler = processor.standardize_data(
            sample_data_with_censoring,
            elements=elements,
            method='standard'
        )
        
        # 验证结果
        assert isinstance(scaled_data, pd.DataFrame)
        assert scaled_data.shape == sample_data_with_censoring[elements].shape
        
        # 验证标准化效果（均值接近0，标准差接近1）
        means = scaled_data.mean()
        stds = scaled_data.std()
        
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-10)
        
        # 验证scaler对象
        assert scaler is not None


class TestFractalAnomalyFilter:
    """测试FractalAnomalyFilter类"""
    
    @pytest.fixture
    def filter(self):
        """创建测试用的分形滤波器实例"""
        return FractalAnomalyFilter(min_samples=30)
    
    @pytest.fixture
    def concentration_data(self):
        """创建测试用的浓度数据"""
        np.random.seed(42)
        
        # 模拟具有分形特征的浓度数据
        background = np.random.lognormal(0, 0.5, 800)  # 背景数据
        anomalies = np.random.lognormal(2, 0.3, 50)    # 异常数据
        
        concentrations = np.concatenate([background, anomalies])
        np.random.shuffle(concentrations)
        
        return concentrations
    
    def test_initialization(self, filter):
        """测试初始化"""
        assert filter.min_samples == 30
        assert filter.ca_results == {}
        assert filter.thresholds == {}
    
    def test_plot_ca_loglog(self, filter, concentration_data):
        """测试C-A双对数图绘制"""
        fig = filter.plot_ca_loglog(
            concentration_data,
            element_name='Test_Element'
        )
        
        # 验证结果
        assert fig is not None
        assert 'Test_Element' in filter.ca_results
        
        # 验证保存的数据
        ca_result = filter.ca_results['Test_Element']
        assert 'concentrations' in ca_result
        assert 'areas' in ca_result
        assert 'log_concentrations' in ca_result
        assert 'log_areas' in ca_result
        assert 'figure' in ca_result
    
    def test_calculate_threshold_interactive(self, filter, concentration_data):
        """测试分形阈值计算"""
        # 先绘制C-A图
        filter.plot_ca_loglog(concentration_data, element_name='Test_Element')
        
        # 计算阈值
        result = filter.calculate_threshold_interactive(
            concentration_data,
            element_name='Test_Element',
            method='knee'
        )
        
        # 验证结果结构
        assert 'threshold' in result
        assert 'method' in result
        assert 'n_anomalies' in result
        assert 'anomaly_percentage' in result
        assert 'method_info' in result
        assert 'anomaly_mask' in result
        assert 'statistics' in result
        
        # 验证数据合理性
        assert result['threshold'] > 0
        assert result['n_anomalies'] >= 0
        assert 0 <= result['anomaly_percentage'] <= 100
        assert len(result['anomaly_mask']) == len(concentration_data)
    
    def test_filter_anomalies(self, filter, concentration_data):
        """测试异常滤波"""
        threshold = np.percentile(concentration_data, 90)
        
        result = filter.filter_anomalies(
            concentration_data,
            threshold=threshold,
            element_name='Test_Element',
            method='binary'
        )
        
        # 验证结果结构
        assert 'threshold' in result
        assert 'method' in result
        assert 'anomaly_mask' in result
        assert 'anomaly_scores' in result
        assert 'anomaly_indices' in result
        assert 'n_anomalies' in result
        assert 'anomaly_percentage' in result
        assert 'anomaly_values' in result
        assert 'background_values' in result
        
        # 验证数据一致性
        assert result['threshold'] == threshold
        assert result['method'] == 'binary'
        assert len(result['anomaly_mask']) == len(concentration_data)
        assert result['n_anomalies'] == np.sum(result['anomaly_mask'])
    
    def test_compare_methods(self, filter, concentration_data):
        """测试方法比较"""
        result = filter.compare_methods(
            concentration_data,
            element_name='Test_Element',
            methods=['knee', 'kmeans', '95th_percentile']
        )
        
        # 验证结果结构
        assert 'results' in result
        assert 'comparison_plot' in result
        assert 'best_method' in result
        
        # 验证方法结果
        for method in ['knee', 'kmeans', '95th_percentile']:
            if method in result['results'] and 'error' not in result['results'][method]:
                assert 'threshold' in result['results'][method]
                assert 'n_anomalies' in result['results'][method]


class TestWeightsOfEvidenceCalculator:
    """测试WeightsOfEvidenceCalculator类"""
    
    @pytest.fixture
    def calculator(self):
        """创建测试用的证据权计算器实例"""
        return WeightsOfEvidenceCalculator(min_cell_size=5)
    
    @pytest.fixture
    def sample_evidence_data(self):
        """创建测试用的证据层数据"""
        np.random.seed(42)
        n_cells = 1000
        
        # 模拟证据层（如断裂密度）
        evidence = np.random.exponential(1, n_cells)
        
        # 模拟训练点（1=矿点，0=非矿点）
        # 高证据值区域有更高的矿点概率
        probabilities = 1 / (1 + np.exp(-2 * (evidence - np.mean(evidence))))
        training_points = np.random.binomial(1, probabilities, n_cells)
        
        return evidence, training_points
    
    def test_initialization(self, calculator):
        """测试初始化"""
        assert calculator.min_cell_size == 5
        assert calculator.woe_results == {}
    
    def test_calculate_studentized_contrast(self, calculator, sample_evidence_data):
        """测试Studentized对比度计算"""
        evidence, training_points = sample_evidence_data
        
        result = calculator.calculate_studentized_contrast(
            evidence,
            training_points,
            evidence_classes=[0.5, 1.0, 2.0, 5.0]
        )
        
        # 验证结果结构
        assert isinstance(result, pd.DataFrame)
        expected_columns = [
            'Class_ID', 'Class_Range', 'Class_Cells', 'Deposits_in_Class',
            'Non_Deposits_in_Class', 'P_A_given_D', 'P_A_given_not_D',
            'W_plus', 'W_minus', 'Contrast', 'Var_W_plus', 'Var_W_minus',
            'Studentized_C', 'Confidence', 'Significant'
        ]
        for col in expected_columns:
            assert col in result.columns
        
        # 验证数据合理性
        assert len(result) > 0
        assert (result['Class_Cells'] >= calculator.min_cell_size).all()
        assert (result['P_A_given_D'] >= 0).all() and (result['P_A_given_D'] <= 1).all()
        assert (result['P_A_given_not_D'] >= 0).all() and (result['P_A_given_not_D'] <= 1).all()
    
    def test_calculate_weights(self, calculator, sample_evidence_data):
        """测试批量证据权计算"""
        evidence, training_points = sample_evidence_data
        
        # 创建多个证据层
        evidence_layers = {
            'evidence_1': evidence,
            'evidence_2': np.random.exponential(0.8, len(evidence)),
            'evidence_3': np.random.exponential(1.2, len(evidence))
        }
        
        results = calculator.calculate_weights(
            evidence_layers,
            training_points
        )
        
        # 验证结果
        assert isinstance(results, dict)
        assert len(results) == 3
        
        for layer_name, result_df in results.items():
            assert isinstance(result_df, pd.DataFrame)
            assert 'Layer_Name' in result_df.columns
            assert (result_df['Layer_Name'] == layer_name).all()
    
    def test_validate_significance(self, calculator, sample_evidence_data):
        """测试统计显著性检验"""
        evidence, training_points = sample_evidence_data
        
        # 先计算证据权
        woe_result = calculator.calculate_studentized_contrast(
            evidence, training_points
        )
        
        # 进行显著性检验
        validation = calculator.validate_significance(woe_result)
        
        # 验证结果结构
        assert 'single_layer' in validation
        assert 'overall' in validation
        
        layer_validation = validation['single_layer']
        assert 'total_classes' in layer_validation
        assert 'significant_classes' in layer_validation
        assert 'mean_studentized_c' in layer_validation
        
        overall = validation['overall']
        assert 'total_layers' in overall
        assert 'total_classes' in overall
        assert 'significant_classes' in overall
        assert 'overall_significance_rate' in overall
    
    def test_plot_woe_results(self, calculator, sample_evidence_data):
        """测试证据权结果绘图"""
        evidence, training_points = sample_evidence_data
        
        # 计算证据权
        woe_result = calculator.calculate_studentized_contrast(
            evidence, training_points
        )
        
        # 绘制结果
        fig = calculator.plot_woe_results(woe_result, plot_type='contrast')
        
        # 验证结果
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])