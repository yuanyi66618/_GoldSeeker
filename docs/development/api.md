# Gold-Seeker API参考

本文档提供Gold-Seeker平台的完整API参考，包括所有类、方法和函数的详细说明。

## 📋 目录

- [核心类](#核心类)
- [智能代理](#智能代理)
- [工具类](#工具类)
- [配置管理](#配置管理)
- [实用工具](#实用工具)
- [异常类](#异常类)

## 🏗️ 核心类

### GoldSeeker

主要的平台入口类，提供完整的地球化学找矿预测功能。

```python
class GoldSeeker:
    """Gold-Seeker地球化学找矿预测智能平台"""
    
    def __init__(self, config: Union[str, Dict] = None, **kwargs):
        """
        初始化Gold-Seeker平台
        
        Args:
            config: 配置文件路径或配置字典
            **kwargs: 其他配置参数
            
        Examples:
            >>> gs = GoldSeeker()
            >>> gs = GoldSeeker("config.yaml")
            >>> gs = GoldSeeker(config={"data": {"format": "csv"}})
        """
```

#### 方法

##### load_data

```python
def load_data(self, data_source: Union[str, pd.DataFrame, gpd.GeoDataFrame], 
               **kwargs) -> gpd.GeoDataFrame:
    """
    加载地球化学数据
    
    Args:
        data_source: 数据源，可以是文件路径、DataFrame或GeoDataFrame
        **kwargs: 加载参数
        
    Returns:
        GeoDataFrame: 加载的地理空间数据
        
    Examples:
        >>> data = gs.load_data("geochemical_data.csv")
        >>> data = gs.load_data("data.shp", encoding="utf-8")
        >>> data = gs.load_data(df, x_col="longitude", y_col="latitude")
    """
```

##### quick_analyze

```python
def quick_analyze(self, data: gpd.GeoDataFrame, target_element: str, 
                  area_name: str = None, **kwargs) -> AnalysisResult:
    """
    快速分析地球化学数据
    
    Args:
        data: 地理空间数据
        target_element: 目标元素
        area_name: 区域名称
        **kwargs: 分析参数
        
    Returns:
        AnalysisResult: 分析结果
        
    Examples:
        >>> results = gs.quick_analyze(data, "Au", "研究区域")
        >>> results = gs.quick_analyze(data, "Au", config={"method": "advanced"})
    """
```

##### execute_workflow

```python
def execute_workflow(self, workflow: WorkflowPlan, data: gpd.GeoDataFrame) -> WorkflowResult:
    """
    执行自定义工作流
    
    Args:
        workflow: 工作流计划
        data: 输入数据
        
    Returns:
        WorkflowResult: 工作流执行结果
        
    Examples:
        >>> workflow = WorkflowPlan("自定义分析")
        >>> results = gs.execute_workflow(workflow, data)
    """
```

##### get_agent

```python
def get_agent(self, agent_name: str) -> BaseAgent:
    """
    获取智能代理实例
    
    Args:
        agent_name: 代理名称
        
    Returns:
        BaseAgent: 代理实例
        
    Examples:
        >>> coordinator = gs.get_agent("Coordinator")
        >>> analyst = gs.get_agent("SpatialAnalyst")
    """
```

##### get_tool

```python
def get_tool(self, tool_name: str) -> BaseTool:
    """
    获取工具实例
    
    Args:
        tool_name: 工具名称
        
    Returns:
        BaseTool: 工具实例
        
    Examples:
        >>> selector = gs.get_tool("GeochemSelector")
        >>> processor = gs.get_tool("GeochemProcessor")
    """
```

## 🤖 智能代理

### CoordinatorAgent

协调代理，负责任务调度和工作流管理。

```python
class CoordinatorAgent(BaseAgent):
    """协调代理 - 任务调度和工作流管理"""
    
    def plan_task(self, task_description: str, data_info: Dict) -> WorkflowPlan:
        """
        规划任务
        
        Args:
            task_description: 任务描述
            data_info: 数据信息
            
        Returns:
            WorkflowPlan: 工作流计划
        """
    
    def coordinate_agents(self, workflow: WorkflowPlan, data: Any) -> Dict[str, Any]:
        """
        协调多个代理执行任务
        
        Args:
            workflow: 工作流计划
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 各代理的执行结果
        """
    
    def monitor_progress(self, workflow_id: str) -> ProgressStatus:
        """
        监控工作流执行进度
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            ProgressStatus: 进度状态
        """
```

### ArchivistAgent

档案代理，负责知识管理和图谱检索。

```python
class ArchivistAgent(BaseAgent):
    """档案代理 - 知识管理和图谱检索"""
    
    def retrieve_knowledge(self, query: str, context: Dict = None) -> List[KnowledgeItem]:
        """
        检索知识
        
        Args:
            query: 查询字符串
            context: 上下文信息
            
        Returns:
            List[KnowledgeItem]: 知识项列表
        """
    
    def build_graph(self, data: Any, domain: str) -> KnowledgeGraph:
        """
        构建知识图谱
        
        Args:
            data: 输入数据
            domain: 领域
            
        Returns:
            KnowledgeGraph: 知识图谱
        """
    
    def query_graph(self, graph: KnowledgeGraph, query: str) -> List[GraphNode]:
        """
        查询知识图谱
        
        Args:
            graph: 知识图谱
            query: 查询字符串
            
        Returns:
            List[GraphNode]: 图节点列表
        """
```

### SpatialAnalystAgent

空间分析代理，负责地球化学数据处理和分析。

```python
class SpatialAnalystAgent(BaseAgent):
    """空间分析代理 - 地球化学数据处理和分析"""
    
    def analyze_geochemical_data(self, data: gpd.GeoDataFrame, 
                                target_element: str) -> AnalysisResult:
        """
        分析地球化学数据
        
        Args:
            data: 地理空间数据
            target_element: 目标元素
            
        Returns:
            AnalysisResult: 分析结果
        """
    
    def process_single_element(self, data: gpd.GeoDataFrame, element: str) -> ElementResult:
        """
        处理单个元素
        
        Args:
            data: 地理空间数据
            element: 元素名称
            
        Returns:
            ElementResult: 元素处理结果
        """
    
    def generate_analysis_report(self, results: AnalysisResult) -> str:
        """
        生成分析报告
        
        Args:
            results: 分析结果
            
        Returns:
            str: 报告内容
        """
```

### ModelerAgent

建模代理，负责机器学习建模和预测。

```python
class ModelerAgent(BaseAgent):
    """建模代理 - 机器学习建模和预测"""
    
    def train_model(self, data: gpd.GeoDataFrame, target_element: str, 
                   model_type: ModelType) -> Model:
        """
        训练模型
        
        Args:
            data: 训练数据
            target_element: 目标元素
            model_type: 模型类型
            
        Returns:
            Model: 训练好的模型
        """
    
    def predict_probability(self, data: gpd.GeoDataFrame, model: Model) -> np.ndarray:
        """
        预测概率
        
        Args:
            data: 预测数据
            model: 训练好的模型
            
        Returns:
            np.ndarray: 预测概率
        """
    
    def validate_model(self, model: Model, test_data: gpd.GeoDataFrame) -> ValidationResult:
        """
        验证模型
        
        Args:
            model: 训练好的模型
            test_data: 测试数据
            
        Returns:
            ValidationResult: 验证结果
        """
```

### CriticAgent

评估代理，负责结果验证和报告生成。

```python
class CriticAgent(BaseAgent):
    """评估代理 - 结果验证和报告生成"""
    
    def validate_logic(self, results: AnalysisResult) -> LogicValidation:
        """
        验证逻辑一致性
        
        Args:
            results: 分析结果
            
        Returns:
            LogicValidation: 逻辑验证结果
        """
    
    def assess_risk(self, results: AnalysisResult) -> RiskAssessment:
        """
        评估风险
        
        Args:
            results: 分析结果
            
        Returns:
            RiskAssessment: 风险评估
        """
    
    def generate_report(self, results: AnalysisResult, validations: List[Validation]) -> ExplorationReport:
        """
        生成报告
        
        Args:
            results: 分析结果
            validations: 验证结果列表
            
        Returns:
            ExplorationReport: 勘探报告
        """
```

## 🔧 工具类

### GeochemSelector

地球化学元素选择工具。

```python
class GeochemSelector(BaseTool):
    """地球化学元素选择工具"""
    
    def select_elements(self, data: gpd.GeoDataFrame, target_element: str, 
                        method: str = "r_mode_clustering") -> List[str]:
        """
        选择相关元素
        
        Args:
            data: 地理空间数据
            target_element: 目标元素
            method: 选择方法
            
        Returns:
            List[str]: 选择的元素列表
        """
    
    def perform_r_mode_analysis(self, data: pd.DataFrame, elements: List[str]) -> ClusterResult:
        """
        执行R型聚类分析
        
        Args:
            data: 数据
            elements: 元素列表
            
        Returns:
            ClusterResult: 聚类结果
        """
    
    def analyze_pca_loadings(self, data: pd.DataFrame, elements: List[str]) -> PCAResult:
        """
        分析PCA载荷
        
        Args:
            data: 数据
            elements: 元素列表
            
        Returns:
            PCAResult: PCA结果
        """
```

### GeochemProcessor

地球化学数据处理工具。

```python
class GeochemProcessor(BaseTool):
    """地球化学数据处理工具"""
    
    def process_data(self, data: gpd.GeoDataFrame, elements: List[str], 
                     censoring_method: str = "substitution") -> gpd.GeoDataFrame:
        """
        处理数据
        
        Args:
            data: 地理空间数据
            elements: 元素列表
            censoring_method: 检测限处理方法
            
        Returns:
            gpd.GeoDataFrame: 处理后的数据
        """
    
    def impute_censored_data(self, data: pd.DataFrame, element: str, 
                            method: str = "substitution") -> pd.Series:
        """
        插补检测限数据
        
        Args:
            data: 数据
            element: 元素
            method: 插补方法
            
        Returns:
            pd.Series: 插补后的数据
        """
    
    def transform_clr(self, data: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
        """
        中心对数比变换
        
        Args:
            data: 数据
            elements: 元素列表
            
        Returns:
            pd.DataFrame: 变换后的数据
        """
```

### FractalAnomalyFilter

分形异常过滤工具。

```python
class FractalAnomalyFilter(BaseTool):
    """分形异常过滤工具"""
    
    def filter_anomalies(self, data: gpd.GeoDataFrame, target_element: str, 
                         method: str = "knee") -> AnomalyResult:
        """
        过滤异常
        
        Args:
            data: 地理空间数据
            target_element: 目标元素
            method: 异常检测方法
            
        Returns:
            AnomalyResult: 异常检测结果
        """
    
    def plot_ca_loglog(self, data: pd.Series, title: str = None) -> plt.Figure:
        """
        绘制C-A双对数图
        
        Args:
            data: 数据
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
    
    def calculate_threshold_interactive(self, data: pd.Series) -> float:
        """
        交互式计算阈值
        
        Args:
            data: 数据
            
        Returns:
            float: 阈值
        """
```

### WeightsOfEvidenceCalculator

证据权计算工具。

```python
class WeightsOfEvidenceCalculator(BaseTool):
    """证据权计算工具"""
    
    def calculate_weights(self, data: gpd.GeoDataFrame, target_element: str, 
                          anomaly_threshold: float) -> WeightResult:
        """
        计算证据权
        
        Args:
            data: 地理空间数据
            target_element: 目标元素
            anomaly_threshold: 异常阈值
            
        Returns:
            WeightResult: 权重计算结果
        """
    
    def calculate_studentized_contrast(self, w_plus: float, w_minus: float, 
                                       w_plus_var: float, w_minus_var: float) -> float:
        """
        计算学生化对比度
        
        Args:
            w_plus: 正权重
            w_minus: 负权重
            w_plus_var: 正权重方差
            w_minus_var: 负权重方差
            
        Returns:
            float: 学生化对比度
        """
    
    def validate_significance(self, studentized_c: float, alpha: float = 0.05) -> bool:
        """
        验证显著性
        
        Args:
            studentized_c: 学生化对比度
            alpha: 显著性水平
            
        Returns:
            bool: 是否显著
        """
```

## ⚙️ 配置管理

### ConfigManager

配置管理器。

```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
    
    def save(self, file_path: str = None) -> None:
        """
        保存配置
        
        Args:
            file_path: 保存路径
        """
    
    def update(self, config: Dict) -> None:
        """
        更新配置
        
        Args:
            config: 配置字典
        """
```

## 🛠️ 实用工具

### 数据验证

```python
def validate_data(data: pd.DataFrame, required_columns: List[str] = None) -> ValidationResult:
    """
    验证数据格式和质量
    
    Args:
        data: 数据
        required_columns: 必需列
        
    Returns:
        ValidationResult: 验证结果
    """

def check_spatial_data(data: gpd.GeoDataFrame) -> SpatialValidationResult:
    """
    检查空间数据
    
    Args:
        data: 地理空间数据
        
    Returns:
        SpatialValidationResult: 空间验证结果
    """
```

### 文件操作

```python
def load_geochemical_data(file_path: str, **kwargs) -> gpd.GeoDataFrame:
    """
    加载地球化学数据
    
    Args:
        file_path: 文件路径
        **kwargs: 加载参数
        
    Returns:
        gpd.GeoDataFrame: 地理空间数据
    """

def save_results(results: AnalysisResult, output_path: str, format: str = "json") -> None:
    """
    保存结果
    
    Args:
        results: 分析结果
        output_path: 输出路径
        format: 输出格式
    """
```

### 可视化

```python
def plot_element_distribution(data: gpd.GeoDataFrame, element: str, 
                            output_path: str = None) -> plt.Figure:
    """
    绘制元素分布图
    
    Args:
        data: 地理空间数据
        element: 元素
        output_path: 输出路径
        
    Returns:
        plt.Figure: 图表对象
    """

def create_interactive_map(data: gpd.GeoDataFrame, value_column: str, 
                           output_path: str = None) -> folium.Map:
    """
    创建交互式地图
    
    Args:
        data: 地理空间数据
        value_column: 值列
        output_path: 输出路径
        
    Returns:
        folium.Map: 地图对象
    """
```

## 🚨 异常类

### GoldSeekerError

基础异常类。

```python
class GoldSeekerError(Exception):
    """Gold-Seeker基础异常类"""
    pass
```

### 数据异常

```python
class DataError(GoldSeekerError):
    """数据相关异常"""
    pass

class DataFormatError(DataError):
    """数据格式异常"""
    pass

class DataValidationError(DataError):
    """数据验证异常"""
    pass
```

### 分析异常

```python
class AnalysisError(GoldSeekerError):
    """分析相关异常"""
    pass

class ElementSelectionError(AnalysisError):
    """元素选择异常"""
    pass

class ProcessingError(AnalysisError):
    """处理异常"""
    pass

class ModelingError(AnalysisError):
    """建模异常"""
    pass
```

### 配置异常

```python
class ConfigError(GoldSeekerError):
    """配置相关异常"""
    pass

class ConfigFileError(ConfigError):
    """配置文件异常"""
    pass

class ConfigValidationError(ConfigError):
    """配置验证异常"""
    pass
```

## 📝 数据类型

### 分析结果

```python
@dataclass
class AnalysisResult:
    """分析结果"""
    selected_elements: List[str]
    processed_data: gpd.GeoDataFrame
    anomalies: AnomalyResult
    weights: WeightResult
    metadata: Dict[str, Any]
    
    def summary(self) -> str:
        """获取结果摘要"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        pass
```

### 工作流计划

```python
@dataclass
class WorkflowPlan:
    """工作流计划"""
    name: str
    description: str
    tasks: List[Task]
    dependencies: Dict[str, List[str]]
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        pass
    
    def validate(self) -> bool:
        """验证工作流"""
        pass
```

### 任务

```python
@dataclass
class Task:
    """任务"""
    name: str
    tool: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    
    def execute(self, data: Any) -> Any:
        """执行任务"""
        pass
```

## 🔌 扩展接口

### 插件接口

```python
class BasePlugin:
    """插件基类"""
    
    def initialize(self, gs_instance: GoldSeeker) -> None:
        """初始化插件"""
        pass
    
    def register_tools(self, gs_instance: GoldSeeker) -> None:
        """注册工具"""
        pass
    
    def register_agents(self, gs_instance: GoldSeeker) -> None:
        """注册代理"""
        pass
    
    def cleanup(self) -> None:
        """清理资源"""
        pass
```

### 工具接口

```python
class BaseTool(ABC):
    """工具基类"""
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """处理数据"""
        pass
    
    def validate_input(self, data: Any) -> bool:
        """验证输入"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        pass
```

### 代理接口

```python
class BaseAgent(ABC):
    """代理基类"""
    
    @abstractmethod
    def execute(self, task: Task, data: Any) -> Any:
        """执行任务"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """获取能力列表"""
        pass
    
    def validate_task(self, task: Task) -> bool:
        """验证任务"""
        pass
```

---

本API参考文档提供了Gold-Seeker平台的完整接口说明，帮助开发者更好地理解和使用平台功能。