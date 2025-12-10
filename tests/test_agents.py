"""
智能体测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from agents.coordinator import CoordinatorAgent
from agents.archivist import ArchivistAgent
from agents.spatial_analyst import SpatialAnalystAgent
from agents.modeler import ModelerAgent
from agents.critic import CriticAgent


class TestCoordinatorAgent:
    """协调智能体测试类"""
    
    def test_coordinator_initialization(self, mock_llm):
        """测试协调智能体初始化"""
        coordinator = CoordinatorAgent(mock_llm)
        
        assert coordinator.llm == mock_llm
        assert hasattr(coordinator, 'workflow_history')
        assert hasattr(coordinator, 'active_tasks')
    
    def test_plan_task_basic(self, mock_llm):
        """测试基本任务规划"""
        coordinator = CoordinatorAgent(mock_llm)
        
        task = {
            'type': 'geochemical_analysis',
            'description': '分析地球化学数据',
            'data_path': 'test_data.csv'
        }
        
        plan = coordinator.plan_task(task)
        
        assert 'steps' in plan
        assert 'agents' in plan
        assert 'timeline' in plan
        assert len(plan['steps']) > 0
    
    def test_coordinate_agents(self, mock_llm):
        """测试智能体协调"""
        coordinator = CoordinatorAgent(mock_llm)
        
        agents = {
            'spatial_analyst': Mock(),
            'modeler': Mock(),
            'critic': Mock()
        }
        
        workflow = {
            'steps': [
                {'agent': 'spatial_analyst', 'task': 'analyze_data'},
                {'agent': 'modeler', 'task': 'train_model'},
                {'agent': 'critic', 'task': 'validate_results'}
            ]
        }
        
        results = coordinator.coordinate_agents(agents, workflow)
        
        assert isinstance(results, dict)
        assert 'spatial_analyst' in results
        assert 'modeler' in results
        assert 'critic' in results
    
    def test_monitor_progress(self, mock_llm):
        """测试进度监控"""
        coordinator = CoordinatorAgent(mock_llm)
        
        active_tasks = {
            'task1': {'status': 'running', 'progress': 0.5},
            'task2': {'status': 'completed', 'progress': 1.0},
            'task3': {'status': 'pending', 'progress': 0.0}
        }
        
        progress = coordinator.monitor_progress(active_tasks)
        
        assert 'overall_progress' in progress
        assert 'completed_tasks' in progress
        assert 'running_tasks' in progress
        assert 'pending_tasks' in progress
    
    def test_handle_failure(self, mock_llm):
        """测试失败处理"""
        coordinator = CoordinatorAgent(mock_llm)
        
        failure_info = {
            'task': 'analyze_data',
            'agent': 'spatial_analyst',
            'error': 'Data loading failed',
            'timestamp': '2025-01-01 10:00:00'
        }
        
        recovery_plan = coordinator.handle_failure(failure_info)
        
        assert 'recovery_steps' in recovery_plan
        assert 'fallback_options' in recovery_plan
        assert 'impact_assessment' in recovery_plan


class TestArchivistAgent:
    """档案管理智能体测试类"""
    
    def test_archivist_initialization(self, mock_llm):
        """测试档案管理智能体初始化"""
        archivist = ArchivistAgent(mock_llm)
        
        assert archivist.llm == mock_llm
        assert hasattr(archivist, 'knowledge_base')
        assert hasattr(archivist, 'graph_database')
    
    def test_retrieve_knowledge(self, mock_llm):
        """测试知识检索"""
        archivist = ArchivistAgent(mock_llm)
        
        query = "卡林型金矿地球化学特征"
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "卡林型金矿通常具有Au-As-Sb-Hg元素组合"
        
        knowledge = archivist.retrieve_knowledge(query)
        
        assert isinstance(knowledge, str)
        assert "卡林型金矿" in knowledge
    
    def test_build_graph(self, mock_llm):
        """测试知识图谱构建"""
        archivist = ArchivistAgent(mock_llm)
        
        documents = [
            "卡林型金矿与Au-As-Sb元素组合相关",
            "斑岩型铜矿与Cu-Mo元素组合相关"
        ]
        
        graph = archivist.build_graph(documents)
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert 'relations' in graph
        assert len(graph['nodes']) > 0
    
    def test_query_graph(self, mock_llm):
        """测试图谱查询"""
        archivist = ArchivistAgent(mock_llm)
        
        # 创建测试图谱
        test_graph = {
            'nodes': [
                {'id': 'Au', 'type': 'element', 'properties': {'name': '金'}},
                {'id': 'As', 'type': 'element', 'properties': {'name': '砷'}},
                {'id': 'carlin_type', 'type': 'deposit', 'properties': {'name': '卡林型'}}
            ],
            'edges': [
                {'source': 'Au', 'target': 'carlin_type', 'relation': 'indicator'},
                {'source': 'As', 'target': 'carlin_type', 'relation': 'indicator'}
            ]
        }
        
        archivist.graph_database = test_graph
        
        results = archivist.query_graph("卡林型金矿的指示元素")
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert any('Au' in str(result) for result in results)
    
    def test_extract_entities(self, mock_llm):
        """测试实体抽取"""
        archivist = ArchivistAgent(mock_llm)
        
        text = "卡林型金矿通常与Au、As、Sb、Hg等元素异常相关"
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = """
        {
            "elements": ["Au", "As", "Sb", "Hg"],
            "deposit_types": ["卡林型金矿"],
            "locations": []
        }
        """
        
        entities = archivist.extract_entities(text)
        
        assert 'elements' in entities
        assert 'deposit_types' in entities
        assert 'Au' in entities['elements']
        assert '卡林型金矿' in entities['deposit_types']


class TestSpatialAnalystAgent:
    """空间分析智能体测试类"""
    
    def test_spatial_analyst_initialization(self, mock_llm, detection_limits):
        """测试空间分析智能体初始化"""
        analyst = SpatialAnalystAgent(mock_llm, detection_limits)
        
        assert analyst.llm == mock_llm
        assert analyst.detection_limits == detection_limits
        assert hasattr(analyst, 'selector')
        assert hasattr(analyst, 'processor')
        assert hasattr(analyst, 'fractal_filter')
        assert hasattr(analyst, 'woe_calculator')
    
    def test_analyze_geochemical_data(self, mock_llm, sample_geochemical_data, detection_limits):
        """测试地球化学数据分析"""
        analyst = SpatialAnalystAgent(mock_llm, detection_limits)
        
        elements = ['Au', 'As', 'Sb', 'Hg']
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "分析完成，识别出Au-As-Sb-Hg元素组合"
        
        result = analyst.analyze_geochemical_data(
            data=sample_geochemical_data,
            elements=elements
        )
        
        assert 'element_selection' in result
        assert 'data_processing' in result
        assert 'anomaly_detection' in result
        assert 'evidence_layers' in result
    
    def test_process_single_element(self, mock_llm, sample_geochemical_data, detection_limits):
        """测试单元素处理"""
        analyst = SpatialAnalystAgent(mock_llm, detection_limits)
        
        element = 'Au'
        
        result = analyst.process_single_element(
            data=sample_geochemical_data,
            element=element
        )
        
        assert 'raw_data' in result
        assert 'processed_data' in result
        assert 'anomalies' in result
        assert 'evidence_layer' in result
    
    def test_generate_analysis_report(self, mock_llm, sample_geochemical_data, detection_limits):
        """测试生成分析报告"""
        analyst = SpatialAnalystAgent(mock_llm, detection_limits)
        
        # 创建模拟结果
        mock_result = {
            'element_selection': {
                'selected_elements': ['Au', 'As', 'Sb', 'Hg'],
                'importance_scores': {'Au': 0.9, 'As': 0.8, 'Sb': 0.7, 'Hg': 0.6}
            },
            'data_processing': {
                'method': 'CLR transformation',
                'outliers_removed': 15
            },
            'anomaly_detection': {
                'method': 'C-A fractal',
                'thresholds': {'Au': 1.5, 'As': 10.0}
            },
            'evidence_layers': {
                'Au': {'contrast': 2.1, 'significant': True},
                'As': {'contrast': 1.8, 'significant': True}
            }
        }
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "# 地球化学分析报告\n\n## 分析结果\n..."
        
        report = analyst.generate_analysis_report(mock_result)
        
        assert isinstance(report, str)
        assert "地球化学分析报告" in report


class TestModelerAgent:
    """建模智能体测试类"""
    
    def test_modeler_initialization(self, mock_llm):
        """测试建模智能体初始化"""
        modeler = ModelerAgent(mock_llm)
        
        assert modeler.llm == mock_llm
        assert hasattr(modeler, 'models')
        assert hasattr(modeler, 'feature_importance')
    
    def test_train_model(self, mock_llm, sample_geochemical_data):
        """测试模型训练"""
        modeler = ModelerAgent(mock_llm)
        
        # 准备训练数据
        features = ['Au', 'As', 'Sb', 'Hg']
        X = sample_geochemical_data[features]
        y = sample_geochemical_data['Is_Deposit']
        
        model_info = modeler.train_model(X, y, model_type='random_forest')
        
        assert 'model' in model_info
        assert 'performance' in model_info
        assert 'feature_importance' in model_info
        assert 'accuracy' in model_info['performance']
    
    def test_predict_probability(self, mock_llm, sample_geochemical_data):
        """测试概率预测"""
        modeler = ModelerAgent(mock_llm)
        
        # 训练模型
        features = ['Au', 'As', 'Sb', 'Hg']
        X = sample_geochemical_data[features]
        y = sample_geochemical_data['Is_Deposit']
        
        model_info = modeler.train_model(X, y)
        
        # 预测
        predictions = modeler.predict_probability(X[:10])  # 前10个样本
        
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_validate_model(self, mock_llm, sample_geochemical_data):
        """测试模型验证"""
        modeler = ModelerAgent(mock_llm)
        
        # 训练模型
        features = ['Au', 'As', 'Sb', 'Hg']
        X = sample_geochemical_data[features]
        y = sample_geochemical_data['Is_Deposit']
        
        model_info = modeler.train_model(X, y)
        
        # 验证
        validation = modeler.validate_model(X, y)
        
        assert 'accuracy' in validation
        assert 'precision' in validation
        assert 'recall' in validation
        assert 'f1_score' in validation
        assert 'roc_auc' in validation
    
    def test_ensemble_models(self, mock_llm, sample_geochemical_data):
        """测试模型集成"""
        modeler = ModelerAgent(mock_llm)
        
        # 准备数据
        features = ['Au', 'As', 'Sb', 'Hg']
        X = sample_geochemical_data[features]
        y = sample_geochemical_data['Is_Deposit']
        
        # 集成多个模型
        ensemble_result = modeler.ensemble_models(
            X, y, 
            model_types=['random_forest', 'logistic_regression']
        )
        
        assert 'ensemble_model' in ensemble_result
        assert 'individual_models' in ensemble_result
        assert 'ensemble_performance' in ensemble_result
        assert len(ensemble_result['individual_models']) == 2


class TestCriticAgent:
    """验证智能体测试类"""
    
    def test_critic_initialization(self, mock_llm):
        """测试验证智能体初始化"""
        critic = CriticAgent(mock_llm)
        
        assert critic.llm == mock_llm
        assert hasattr(critic, 'validation_rules')
        assert hasattr(critic, 'risk_assessment')
    
    def test_validate_logic(self, mock_llm):
        """测试逻辑验证"""
        critic = CriticAgent(mock_llm)
        
        # 创建分析结果
        analysis_result = {
            'element_selection': {'selected_elements': ['Au', 'As', 'Sb', 'Hg']},
            'evidence_layers': {
                'Au': {'contrast': 2.1, 'significant': True},
                'As': {'contrast': 1.8, 'significant': True}
            }
        }
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "逻辑验证通过，分析结果合理"
        
        validation = critic.validate_logic(analysis_result)
        
        assert 'is_valid' in validation
        assert 'issues' in validation
        assert 'recommendations' in validation
    
    def test_assess_risk(self, mock_llm):
        """测试风险评估"""
        critic = CriticAgent(mock_llm)
        
        # 创建预测结果
        prediction_result = {
            'target_areas': [
                {'location': 'Area1', 'probability': 0.85, 'confidence': 0.7},
                {'location': 'Area2', 'probability': 0.65, 'confidence': 0.5}
            ],
            'model_performance': {'accuracy': 0.82, 'roc_auc': 0.88}
        }
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "风险评估完成，建议优先勘探Area1"
        
        risk_assessment = critic.assess_risk(prediction_result)
        
        assert 'overall_risk' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'mitigation_strategies' in risk_assessment
    
    def test_generate_report(self, mock_llm):
        """测试报告生成"""
        critic = CriticAgent(mock_llm)
        
        # 创建完整分析结果
        complete_result = {
            'analysis': {
                'element_selection': {'selected_elements': ['Au', 'As', 'Sb', 'Hg']},
                'evidence_layers': {'Au': {'contrast': 2.1}}
            },
            'modeling': {
                'performance': {'accuracy': 0.82, 'roc_auc': 0.88},
                'predictions': {'target_areas': ['Area1', 'Area2']}
            },
            'validation': {
                'is_valid': True,
                'risk_assessment': {'overall_risk': 'medium'}
            }
        }
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "# 金矿找矿预测报告\n\n## 执行摘要\n..."
        
        report = critic.generate_report(complete_result)
        
        assert isinstance(report, str)
        assert "金矿找矿预测报告" in report
    
    def test_expert_review(self, mock_llm):
        """测试专家评审"""
        critic = CriticAgent(mock_llm)
        
        # 创建分析结果
        analysis_result = {
            'methodology': 'C-A fractal + Weights of Evidence',
            'key_findings': ['Au-As-Sb元素组合', '高异常区域识别'],
            'limitations': ['数据覆盖不完整', '训练样本较少']
        }
        
        # 模拟LLM响应
        mock_llm.invoke.return_value = "专家评审：方法合理，结果可信，建议补充数据"
        
        review = critic.expert_review(analysis_result)
        
        assert 'overall_rating' in review
        assert 'strengths' in review
        assert 'weaknesses' in review
        assert 'recommendations' in review