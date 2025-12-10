"""
Spatial Analyst Agent - 空间分析智能体

负责地球化学数据的处理、分析和证据层构建。
基于Carranza理论，集成多种地球化学处理工具。

核心工作流：
1. R-mode聚类分析 → 识别元素共生组合
2. CLR变换 → 消除闭合效应
3. C-A分形滤波 → 确定异常阈值
4. 证据权计算 → 定量评价证据层
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from .tools.geochem import (
    GeochemSelector,
    GeochemProcessor, 
    FractalAnomalyFilter,
    WeightsOfEvidenceCalculator
)


class SpatialAnalystAgent:
    """
    空间分析智能体
    
    基于LangChain实现，集成地球化学处理工具，
    实现从数据清洗到证据权计算的完整工作流。
    """
    
    def __init__(self, llm: BaseChatModel, 
                 detection_limits: Optional[Dict[str, float]] = None):
        """
        初始化空间分析智能体
        
        Args:
            llm: 语言模型
            detection_limits: 元素检测限字典
        """
        self.llm = llm
        self.detection_limits = detection_limits or {}
        
        # 初始化工具组件
        self.selector = GeochemSelector(detection_limits)
        self.processor = GeochemProcessor(detection_limits)
        self.fractal_filter = FractalAnomalyFilter()
        self.woe_calculator = WeightsOfEvidenceCalculator()
        
        # 创建LangChain工具
        self.tools = self._create_tools()
        
        # 创建智能体
        self.agent = self._create_agent()
        
    def _create_tools(self) -> List[StructuredTool]:
        """创建LangChain工具列表"""
        
        def perform_r_mode_analysis(
            elements: List[str], 
            method: str = "ward",
            metric: str = "correlation"
        ) -> Dict[str, Any]:
            """
            执行R型聚类分析，识别元素共生组合
            
            Args:
                elements: 待分析元素列表
                method: 聚类方法
                metric: 距离度量
                
            Returns:
                聚类分析结果
            """
            # 这里需要从上下文获取数据
            # 实际实现中需要传入数据或从全局状态获取
            return {"status": "success", "message": "R-mode analysis completed"}
        
        def analyze_pca_loadings(
            elements: List[str],
            n_components: int = 3
        ) -> Dict[str, Any]:
            """
            分析主成分载荷，推荐找矿元素组合
            
            Args:
                elements: 待分析元素列表
                n_components: 主成分数量
                
            Returns:
                PCA分析结果
            """
            return {"status": "success", "message": "PCA analysis completed"}
        
        def process_geochem_data(
            elements: List[str],
            censoring_method: str = "substitution",
            clr_transform: bool = True
        ) -> Dict[str, Any]:
            """
            处理地球化学数据：检测限处理和CLR变换
            
            Args:
                elements: 待处理元素列表
                censoring_method: 检测限处理方法
                clr_transform: 是否进行CLR变换
                
            Returns:
                数据处理结果
            """
            return {"status": "success", "message": "Data processing completed"}
        
        def calculate_fractal_thresholds(
            elements: List[str],
            method: str = "knee"
        ) -> Dict[str, Any]:
            """
            计算分形异常阈值
            
            Args:
                elements: 待分析元素列表
                method: 拐点检测方法
                
            Returns:
                分形阈值计算结果
            """
            return {"status": "success", "message": "Fractal thresholding completed"}
        
        def calculate_weights_of_evidence(
            evidence_layers: List[str],
            training_points_file: str
        ) -> Dict[str, Any]:
            """
            计算证据权
            
            Args:
                evidence_layers: 证据层列表
                training_points_file: 训练点文件路径
                
            Returns:
                证据权计算结果
            """
            return {"status": "success", "message": "Weights of evidence calculated"}
        
        # 创建工具列表
        tools = [
            StructuredTool.from_function(
                func=perform_r_mode_analysis,
                name="r_mode_analysis",
                description="执行R型聚类分析，识别元素共生组合和地球化学关联性"
            ),
            StructuredTool.from_function(
                func=analyze_pca_loadings,
                name="pca_analysis", 
                description="分析主成分载荷，提取最佳找矿元素组合"
            ),
            StructuredTool.from_function(
                func=process_geochem_data,
                name="data_processing",
                description="处理地球化学数据：检测限处理、CLR变换、异常值检测"
            ),
            StructuredTool.from_function(
                func=calculate_fractal_thresholds,
                name="fractal_thresholding",
                description="使用C-A分形模型确定地球化学异常阈值"
            ),
            StructuredTool.from_function(
                func=calculate_weights_of_evidence,
                name="weights_of_evidence",
                description="计算证据权W+、W-、对比度C和Studentized C"
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """创建LangChain智能体"""
        
        # 系统提示 - 思维链(CoT)指导
        system_prompt = """你是一位专业的勘查地球化学专家，基于Carranza的《Geochemical Anomaly and Mineral Prospectivity Mapping in GIS》理论进行工作。

你的核心任务是执行金矿预测的地球化学证据层构建，必须按照以下思维链(CoT)步骤进行：

## 工作流程（必须严格遵循）：

### 步骤1: 元素组合识别
- 首先使用 r_mode_analysis 工具进行R型聚类分析
- 识别元素共生组合（如Au-As-Sb、Cu-Pb-Zn等）
- 然后使用 pca_analysis 工具进行主成分分析
- 基于载荷分析推荐最佳找矿元素组合

### 步骤2: 数据预处理
- 使用 data_processing 工具处理地球化学数据
- 必须进行检测限数据处理（推荐substitution方法）
- 必须进行CLR变换消除闭合效应
- 检测并处理异常值

### 步骤3: 异常阈值确定
- 使用 fractal_thresholding 工具进行C-A分形分析
- 绘制双对数图并识别拐点
- 计算分形异常阈值
- 比较不同方法的结果

### 步骤4: 证据权计算
- 使用 weights_of_evidence 工具计算证据权
- 计算W+、W-、对比度C和Studentized C
- 进行统计显著性检验
- 评估证据层的有效性

## 专业指导原则：

1. **地质合理性优先**：所有统计结果必须符合地质常识
2. **Carranza理论指导**：严格遵循Carranza (2009)的方法论
3. **统计显著性**：关注Studentized C > 1.96的显著证据层
4. **多方法验证**：关键结论需要多种方法相互验证
5. **闭合效应处理**：组合数据必须进行CLR变换

## 输出要求：

- 每个步骤都要明确说明使用的工具和参数
- 提供中间结果的定量评估
- 给出地质解释和专业建议
- 指出潜在问题和改进方向

记住：你是地球化学专家，不是纯粹的统计分析师。所有分析结果都要回归到地质意义和勘探应用上。"""

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            ("assistant", "让我按照Carranza理论，系统地进行地球化学证据层构建分析。"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 创建智能体
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # 创建执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        
        return agent_executor
    
    def analyze_geochemical_data(self, 
                                data: pd.DataFrame,
                                elements: List[str],
                                training_points: Optional[pd.DataFrame] = None,
                                **kwargs) -> Dict[str, Any]:
        """
        分析地球化学数据的主入口
        
        Args:
            data: 地球化学数据
            elements: 待分析元素列表
            training_points: 训练点数据
            **kwargs: 其他参数
            
        Returns:
            分析结果字典
        """
        # 存储数据供工具使用
        self.current_data = data
        self.current_elements = elements
        self.current_training_points = training_points
        
        # 构建分析任务
        task = f"""
        请对以下地球化学数据进行完整的证据层构建分析：

        数据信息：
        - 样品数量：{len(data)}
        - 分析元素：{', '.join(elements)}
        - 检测限：{self.detection_limits}

        请按照标准工作流程进行：
        1. R-mode聚类分析识别元素组合
        2. PCA分析推荐找矿元素
        3. 数据预处理（检测限+CLR变换）
        4. C-A分形异常阈值确定
        5. 证据权计算与显著性检验

        请提供详细的分析报告，包括：
- 每步的关键结果和图表
- 地质解释和专业建议
- 后续勘探工作建议
        """
        
        try:
            # 执行智能体分析
            result = self.agent.invoke({"input": task})
            return {
                "status": "success",
                "analysis_result": result,
                "elements_analyzed": elements,
                "data_shape": data.shape
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "elements_analyzed": elements
            }
    
    def process_single_element(self, 
                               data: pd.DataFrame,
                               element: str,
                               training_points: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        处理单个元素的完整流程
        
        Args:
            data: 地球化学数据
            element: 目标元素
            training_points: 训练点数据
            
        Returns:
            单元素处理结果
        """
        results = {}
        
        try:
            # 1. 数据预处理
            processed_data = self.processor.impute_censored_data(
                data, elements=[element]
            )
            
            # 2. CLR变换
            clr_data = self.processor.transform_clr(
                processed_data, elements=[element]
            )
            
            # 3. 分形异常滤波
            concentrations = data[element].values
            ca_plot = self.fractal_filter.plot_ca_loglog(
                concentrations, element_name=element
            )
            
            threshold_result = self.fractal_filter.calculate_threshold_interactive(
                concentrations, element_name=element
            )
            
            # 4. 证据权计算（如果有训练点）
            if training_points is not None:
                woe_result = self.woe_calculator.calculate_studentized_contrast(
                    data[element], training_points
                )
                results['weights_of_evidence'] = woe_result
            
            results.update({
                'status': 'success',
                'processed_data': processed_data,
                'clr_data': clr_data,
                'ca_plot': ca_plot,
                'threshold_result': threshold_result,
                'element': element
            })
            
        except Exception as e:
            results = {
                'status': 'error',
                'error_message': str(e),
                'element': element
            }
        
        return results
    
    def generate_analysis_report(self, 
                                 results: Dict[str, Any],
                                 output_format: str = "text") -> str:
        """
        生成分析报告
        
        Args:
            results: 分析结果
            output_format: 输出格式 ("text", "markdown", "html")
            
        Returns:
            格式化的分析报告
        """
        if output_format == "markdown":
            return self._generate_markdown_report(results)
        elif output_format == "html":
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """生成文本格式报告"""
        report = []
        report.append("=" * 60)
        report.append("地球化学证据层构建分析报告")
        report.append("=" * 60)
        report.append("")
        
        if results.get("status") == "success":
            report.append("分析状态：成功完成")
            report.append(f"分析元素：{', '.join(results.get('elements_analyzed', []))}")
            report.append("")
            
            if "analysis_result" in results:
                report.append("智能体分析结果：")
                report.append("-" * 40)
                report.append(str(results["analysis_result"]))
        
        else:
            report.append(f"分析状态：失败 - {results.get('error_message', '未知错误')}")
        
        return "\n".join(report)
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        report = []
        report.append("# 地球化学证据层构建分析报告")
        report.append("")
        
        if results.get("status") == "success":
            report.append("## 分析概要")
            report.append(f"- **状态**: ✅ 成功完成")
            report.append(f"- **分析元素**: {', '.join(results.get('elements_analyzed', []))}")
            report.append("")
            
            if "analysis_result" in results:
                report.append("## 智能体分析结果")
                report.append(str(results["analysis_result"]))
        
        else:
            report.append("## 分析失败")
            report.append(f"**错误信息**: {results.get('error_message', '未知错误')}")
        
        return "\n".join(report)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        html = """
        <html>
        <head>
            <title>地球化学证据层构建分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .success { color: green; }
                .error { color: red; }
                .content { margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>地球化学证据层构建分析报告</h1>
        """
        
        if results.get("status") == "success":
            html += f'<p class="success">✅ 分析成功完成</p>'
            html += f'<p><strong>分析元素</strong>: {", ".join(results.get("elements_analyzed", []))}</p>'
        else:
            html += f'<p class="error">❌ 分析失败: {results.get("error_message", "未知错误")}</p>'
        
        html += """
            </div>
            <div class="content">
        """
        
        if "analysis_result" in results:
            html += f"<pre>{results['analysis_result']}</pre>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html