"""
集成测试

测试各组件之间的集成和端到端工作流。
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from agents import (
    CoordinatorAgent, ArchivistAgent, SpatialAnalystAgent, 
    ModelerAgent, CriticAgent
)
from agents.config import load_config
from agents.tools.geochem import (
    GeochemSelector, GeochemProcessor, FractalAnomalyFilter, 
    WeightsOfEvidenceCalculator
)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """端到端工作流测试类"""
    
    def test_complete_geochemical_workflow(self, mock_llm, sample_geochemical_data, detection_limits, tmp_path):
        """测试完整的地球化学分析工作流"""
        
        # 1. 初始化所有智能体
        coordinator = CoordinatorAgent(mock_llm)
        archivist = ArchivistAgent(mock_llm)
        analyst = SpatialAnalystAgent(mock_llm, detection_limits)
        modeler = ModelerAgent(mock_llm)
        critic = CriticAgent(mock_llm)
        
        # 2. 定义分析任务
        task = {
            'type': 'geochemical_prospecting',
            'description': '卡林型金矿地球化学找矿预测',
            'data': sample_geochemical_data,
            'elements': ['Au', 'As', 'Sb', 'Hg'],
            'target_mineral': 'gold',
            'deposit_type': 'carlin_type'
        }
        
        # 3. 任务规划
        workflow_plan = coordinator.plan_task(task)
        assert 'steps' in workflow_plan
        assert len(workflow_plan['steps']) > 0
        
        # 4. 知识检索
        knowledge = archivist.retrieve_knowledge("卡林型金矿地球化学特征")
        assert isinstance(knowledge, str)
        
        # 5. 地球化学数据分析
        analysis_result = analyst.analyze_geochemical_data(
            data=sample_geochemical_data,
            elements=['Au', 'As', 'Sb', 'Hg'],
            training_points=sample_geochemical_data[sample_geochemical_data['Is_Deposit'] == 1]
        )
        
        assert 'element_selection' in analysis_result
        assert 'evidence_layers' in analysis_result
        
        # 6. 建模
        features = ['Au', 'As', 'Sb', 'Hg']
        X = sample_geochemical_data[features]
        y = sample_geochemical_data['Is_Deposit']
        
        model_info = modeler.train_model(X, y, model_type='random_forest')
        predictions = modeler.predict_probability(X)
        
        assert 'model' in model_info
        assert len(predictions) == len(X)
        
        # 7. 验证和报告
        validation_result = critic.validate_logic(analysis_result)
        risk_assessment = critic.assess_risk({'predictions': predictions})
        final_report = critic.generate_report({
            'analysis': analysis_result,
            'modeling': model_info,
            'validation': validation_result
        })
        
        assert isinstance(final_report, str)
        assert len(final_report) > 0
        
        # 8. 保存结果
        output_dir = tmp_path / "workflow_output"
        output_dir.mkdir(exist_ok=True)
        
        # 保存分析结果
        analysis_file = output_dir / "analysis_result.json"
        import json
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存报告
        report_file = output_dir / "final_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        assert analysis_file.exists()
        assert report_file.exists()
    
    def test_multi_agent_coordination(self, mock_llm, sample_geochemical_data, detection_limits):
        """测试多智能体协调"""
        
        # 初始化智能体
        coordinator = CoordinatorAgent(mock_llm)
        
        # 创建模拟智能体
        agents = {
            'spatial_analyst': MockSpatialAnalyst(),
            'modeler': MockModeler(),
            'critic': MockCritic()
        }
        
        # 定义工作流
        workflow = {
            'steps': [
                {'agent': 'spatial_analyst', 'task': 'analyze_data', 'priority': 1},
                {'agent': 'modeler', 'task': 'train_model', 'priority': 2},
                {'agent': 'critic', 'task': 'validate_results', 'priority': 3}
            ]
        }
        
        # 执行协调
        results = coordinator.coordinate_agents(agents, workflow)
        
        assert 'spatial_analyst' in results
        assert 'modeler' in results
        assert 'critic' in results
        
        # 验证执行顺序
        assert results['spatial_analyst']['status'] == 'completed'
        assert results['modeler']['status'] == 'completed'
        assert results['critic']['status'] == 'completed'
    
    def test_error_handling_and_recovery(self, mock_llm):
        """测试错误处理和恢复"""
        
        coordinator = CoordinatorAgent(mock_llm)
        
        # 模拟失败信息
        failure_info = {
            'task': 'data_loading',
            'agent': 'spatial_analyst',
            'error': 'File not found: data.csv',
            'timestamp': '2025-01-01 10:00:00',
            'context': {
                'file_path': 'data.csv',
                'expected_format': 'csv'
            }
        }
        
        # 处理失败
        recovery_plan = coordinator.handle_failure(failure_info)
        
        assert 'recovery_steps' in recovery_plan
        assert 'fallback_options' in recovery_plan
        assert 'impact_assessment' in recovery_plan
        
        # 验证恢复步骤
        recovery_steps = recovery_plan['recovery_steps']
        assert len(recovery_steps) > 0
        assert any('check_file_path' in step for step in recovery_steps)


@pytest.mark.integration
class TestToolIntegration:
    """工具集成测试类"""
    
    def test_geochem_tool_pipeline(self, sample_geochemical_data, detection_limits):
        """测试地球化学工具管道"""
        
        # 1. 元素选择
        selector = GeochemSelector(detection_limits)
        elements = ['Au', 'As', 'Sb', 'Hg', 'Cu', 'Pb', 'Zn']
        
        r_mode_result = selector.perform_r_mode_analysis(sample_geochemical_data, elements)
        pca_result = selector.analyze_pca_loadings(sample_geochemical_data, elements)
        importance_result = selector.rank_element_importance(r_mode_result, pca_result)
        
        assert 'clusters' in r_mode_result
        assert 'loadings' in pca_result
        assert 'ranked_elements' in importance_result
        
        # 2. 数据处理
        processor = GeochemProcessor(detection_limits)
        selected_elements = importance_result['ranked_elements'][:4]  # 前4个重要元素
        
        processed_data = processor.transform_clr(sample_geochemical_data, selected_elements)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(sample_geochemical_data)
        
        # 3. 分形异常滤波
        fractal_filter = FractalAnomalyFilter()
        
        anomaly_results = {}
        for element in selected_elements:
            if element in processed_data.columns:
                result = fractal_filter.calculate_threshold_interactive(
                    processed_data[element], element
                )
                anomaly_results[element] = result
        
        assert len(anomaly_results) > 0
        
        # 4. 证据权计算
        woe_calculator = WeightsOfEvidenceCalculator()
        
        woe_results = {}
        for element, anomaly_result in anomaly_results.items():
            # 创建二元证据层
            threshold = anomaly_result['threshold']
            evidence_layer = (processed_data[element] > threshold).astype(int)
            
            # 计算证据权
            training_points = sample_geochemical_data['Is_Deposit']
            woe_result = woe_calculator.calculate_weights(
                evidence_layer, training_points
            )
            woe_results[element] = woe_result
        
        assert len(woe_results) > 0
        
        # 验证管道一致性
        for element in woe_results:
            assert 'weights' in woe_results[element]
            assert 'contrast' in woe_results[element]
            assert 'studentized_contrast' in woe_results[element]
    
    def test_data_flow_consistency(self, sample_geochemical_data, detection_limits):
        """测试数据流一致性"""
        
        # 原始数据
        original_data = sample_geochemical_data.copy()
        original_shape = original_data.shape
        
        # 处理步骤
        processor = GeochemProcessor(detection_limits)
        elements = ['Au', 'As', 'Sb', 'Hg']
        
        # 1. 检测限处理
        censored_data = processor.impute_censored_data(original_data, elements)
        assert censored_data.shape == original_shape
        
        # 2. CLR变换
        clr_data = processor.transform_clr(censored_data, elements)
        assert clr_data.shape[0] == original_shape[0]
        assert len(clr_data.columns) == len(elements)
        
        # 3. 异常检测
        outliers = processor.detect_outliers(clr_data, method='iqr')
        assert len(outliers) == len(clr_data)
        
        # 4. 数据标准化
        standardized_data = processor.standardize_data(clr_data)
        assert standardized_data.shape == clr_data.shape
        
        # 验证数据完整性
        assert not standardized_data.isnull().any().any()
        assert np.isfinite(standardized_data.values).all()


@pytest.mark.integration
class TestConfigIntegration:
    """配置集成测试类"""
    
    def test_config_driven_workflow(self, test_config, sample_geochemical_data, tmp_path):
        """测试配置驱动的工作流"""
        
        # 创建配置文件
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)
        
        # 加载配置
        config_manager = load_config(config_file)
        
        # 验证配置应用
        assert config_manager.get("global.project_name") == "Gold-Seeker-Test"
        assert config_manager.get("geochemistry.censoring_method") == "substitution"
        
        # 使用配置创建工具
        detection_limits = config_manager.get_detection_limits()
        selector = GeochemSelector(detection_limits)
        
        # 验证配置参数应用
        elements = ['Au', 'As', 'Sb', 'Hg']
        r_mode_result = selector.perform_r_mode_analysis(sample_geochemical_data, elements)
        
        assert 'clusters' in r_mode_result
        assert len(r_mode_result['clusters']) > 0
    
    def test_output_configuration(self, test_config, tmp_path):
        """测试输出配置"""
        
        # 设置输出目录
        output_dir = tmp_path / "configured_output"
        test_config["output"]["output_dir"] = str(output_dir)
        
        # 创建配置管理器
        from agents.config import ConfigManager
        manager = ConfigManager()
        manager.config = test_config
        
        # 获取输出目录
        configured_output_dir = manager.get_output_dir()
        
        assert configured_output_dir == output_dir
        assert configured_output_dir.exists()


# 模拟类用于测试
class MockSpatialAnalyst:
    """模拟空间分析智能体"""
    
    def analyze_data(self, data, elements, training_points=None):
        return {
            'status': 'completed',
            'element_selection': {'selected_elements': elements},
            'evidence_layers': {elem: {'contrast': 1.5} for elem in elements}
        }


class MockModeler:
    """模拟建模智能体"""
    
    def train_model(self, X, y, model_type='random_forest'):
        return {
            'status': 'completed',
            'model_type': model_type,
            'performance': {'accuracy': 0.85}
        }


class MockCritic:
    """模拟验证智能体"""
    
    def validate_results(self, analysis_result, modeling_result):
        return {
            'status': 'completed',
            'is_valid': True,
            'confidence': 0.9
        }