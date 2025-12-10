# Gold-Seeker API 参考文档

本文档提供Gold-Seeker地球化学找矿预测智能平台的完整API参考，包括所有类、方法和函数的详细说明。

## 📋 目录

- [核心模块](#核心模块)
- [代理模块](#代理模块)
- [工具模块](#工具模块)
- [配置模块](#配置模块)
- [实用工具](#实用工具)
- [CLI接口](#cli接口)
- [类型定义](#类型定义)

## 🏗️ 核心模块

### GoldSeeker类

主要的平台入口类，提供高级API接口。

```python
class GoldSeeker:
    """Gold-Seeker地球化学找矿预测平台主类"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化Gold-Seeker平台
        
        Args:
            config_path: 配置文件路径
            **kwargs: 额外配置参数
        
        Example:
            >>> gs = GoldSeeker(config_path="config/my_config.yaml")
            >>> gs = GoldSeeker(data_dir="./data", n_jobs=4)
        """
    
    def quick_analyze(self, 
                     data: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                     target_element: str,
                     **kwargs) -> AnalysisResult:
        """
        快速分析接口
        
        Args:
            data: 输入数据（文件路径或DataFrame）
            target_element: 目标元素（如"Au"）
            **kwargs: 额外参数
        
        Returns:
            AnalysisResult: 分析结果对象
        
        Example:
            >>> result = gs.quick_analyze("data/geochem.csv", "Au")
            >>> print(result.summary)
        """
    
    def full_workflow(self, 
                     data: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                     target_element: str,
                     workflow_config: Optional[Dict] = None) -> WorkflowResult:
        """
        完整工作流程
        
        Args:
            data: 输入数据
            target_element: 目标元素
            workflow_config: 工作流配置
        
        Returns:
            WorkflowResult: 工作流结果
        
        Example:
            >>> config = {"feature_selection": "r_mode", "anomaly_method": "c_a"}
            >>> result = gs.full_workflow("data.csv", "Au", config)
        """
    
    def batch_analyze(self, 
                      data_list: List[Union[str, pd.DataFrame]],
                      target_elements: List[str],
                      **kwargs) -> List[AnalysisResult]:
        """
        批量分析
        
        Args:
            data_list: 数据列表
            target_elements: 目标元素列表
            **kwargs: 额外参数
        
        Returns:
            List[AnalysisResult]: 分析结果列表
        """
```

### AnalysisResult类

分析结果容器类。

```python
@dataclass
class AnalysisResult:
    """分析结果数据类"""
    
    # 输入信息
    input_data: pd.DataFrame
    target_element: str
    config: Dict[str, Any]
    
    # 特征选择结果
    selected_features: List[str]
    feature_importance: pd.Series
    
    # 数据处理结果
    processed_data: pd.DataFrame
    outliers_removed: int
    
    # 异常检测结果
    anomaly_threshold: float
    anomaly_points: pd.DataFrame
    
    # 权重分析结果
    weights: pd.DataFrame
    contrast: pd.Series
    studentized_contrast: pd.Series
    
    # 统计信息
    statistics: Dict[str, Any]
    
    # 元数据
    timestamp: datetime
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
    
    def to_json(self) -> str:
        """转换为JSON格式"""
    
    def save(self, filepath: str) -> None:
        """保存结果到文件"""
    
    def plot(self, plot_type: str = "summary", **kwargs) -> None:
        """绘制结果图表"""
    
    def get_summary(self) -> str:
        """获取结果摘要"""
```

## 🤖 代理模块

### CoordinatorAgent

任务协调代理，负责工作流程管理。

```python
class CoordinatorAgent(BaseAgent):
    """任务协调代理"""
    
    def plan_task(self, task_description: str, context: Dict) -> WorkflowPlan:
        """
        规划任务
        
        Args:
            task_description: 任务描述
            context: 上下文信息
        
        Returns:
            WorkflowPlan: 工作流计划
        """
    
    def coordinate_agents(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """
        协调各代理执行任务
        
        Args:
            plan: 工作流计划
        
        Returns:
            Dict[str, Any]: 执行结果
        """
    
    def monitor_progress(self, task_id: str) -> TaskStatus:
        """
        监控任务进度
        
        Args:
            task_id: 任务ID
        
        Returns:
            TaskStatus: 任务状态
        """
```

### ArchivistAgent

知识管理代理，负责知识检索和图谱构建。

```python
class ArchivistAgent(BaseAgent):
    """知识管理代理"""
    
    def retrieve_knowledge(self, query: str, domain: str) -> List[KnowledgeItem]:
        """
        检索知识
        
        Args:
            query: 查询字符串
            domain: 知识域
        
        Returns:
            List[KnowledgeItem]: 知识项列表
        """
    
    def build_graph(self, entities: List[Entity], relations: List[Relation]) -> KnowledgeGraph:
        """
        构建知识图谱
        
        Args:
            entities: 实体列表
            relations: 关系列表
        
        Returns:
            KnowledgeGraph: 知识图谱
        """
    
    def query_graph(self, graph: KnowledgeGraph, query: GraphQuery) -> List[GraphResult]:
        """
        查询知识图谱
        
        Args:
            graph: 知识图谱
            query: 图查询
        
        Returns:
            List[GraphResult]: 查询结果
        """
```

### ModelerAgent

建模代理，负责机器学习模型训练和预测。

```python
class ModelerAgent(BaseAgent):
    """建模代理"""
    
    def train_model(self, 
                   training_data: TrainingData,
                   model_type: ModelType,
                   hyperparameters: Optional[Dict] = None) -> TrainedModel:
        """
        训练模型
        
        Args:
            training_data: 训练数据
            model_type: 模型类型
            hyperparameters: 超参数
        
        Returns:
            TrainedModel: 训练好的模型
        """
    
    def predict_probability(self, 
                           model: TrainedModel,
                           evidence_layers: List[EvidenceLayer]) -> np.ndarray:
        """
        预测成矿概率
        
        Args:
            model: 训练好的模型
            evidence_layers: 证据图层
        
        Returns:
            np.ndarray: 预测概率
        """
    
    def validate_model(self, 
                      model: TrainedModel,
                      validation_data: ValidationData) -> ModelValidation:
        """
        验证模型
        
        Args:
            model: 训练好的模型
            validation_data: 验证数据
        
        Returns:
            ModelValidation: 验证结果
        """
```

### CriticAgent

评估代理，负责结果验证和报告生成。

```python
class CriticAgent(BaseAgent):
    """评估代理"""
    
    def validate_logic(self, results: Dict[str, Any]) -> ValidationResult:
        """
        验证逻辑一致性
        
        Args:
            results: 分析结果
        
        Returns:
            ValidationResult: 验证结果
        """
    
    def assess_risk(self, predictions: np.ndarray, confidence: np.ndarray) -> RiskAssessment:
        """
        评估风险
        
        Args:
            predictions: 预测结果
            confidence: 置信度
        
        Returns:
            RiskAssessment: 风险评估
        """
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       template: Optional[str] = None) -> ExplorationReport:
        """
        生成报告
        
        Args:
            results: 分析结果
            template: 报告模板
        
        Returns:
            ExplorationReport: 勘探报告
        """
```

### SpatialAnalystAgent

空间分析代理，集成LangChain进行智能分析。

```python
class SpatialAnalystAgent(BaseAgent):
    """空间分析代理"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        """
        初始化空间分析代理
        
        Args:
            llm: 语言模型实例
            **kwargs: 额外参数
        """
    
    def analyze_geochemical_data(self, 
                                data: pd.DataFrame,
                                target_element: str,
                                analysis_type: str = "full") -> Dict[str, Any]:
        """
        分析地球化学数据
        
        Args:
            data: 地球化学数据
            target_element: 目标元素
            analysis_type: 分析类型
        
        Returns:
            Dict[str, Any]: 分析结果
        """
    
    def process_single_element(self, 
                              data: pd.DataFrame,
                              element: str) -> Dict[str, Any]:
        """
        处理单个元素
        
        Args:
            data: 数据
            element: 元素名称
        
        Returns:
            Dict[str, Any]: 处理结果
        """
    
    def generate_analysis_report(self, results: Dict[str, Any]) -> str:
        """
        生成分析报告
        
        Args:
            results: 分析结果
        
        Returns:
            str: 报告文本
        """
```

## 🛠️ 工具模块

### GeochemSelector

地球化学特征选择工具。

```python
class GeochemSelector:
    """地球化学特征选择器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化特征选择器
        
        Args:
            config: 配置参数
        """
    
    def perform_r_mode_analysis(self, 
                               data: pd.DataFrame,
                               target_element: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        执行R型聚类分析
        
        Args:
            data: 输入数据
            target_element: 目标元素
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: 相关性矩阵和选择的特征
        """
    
    def analyze_pca_loadings(self, 
                            data: pd.DataFrame,
                            n_components: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        分析PCA载荷
        
        Args:
            data: 输入数据
            n_components: 主成分数量
        
        Returns:
            Tuple[np.ndarray, List[str]]: 载荷矩阵和重要特征
        """
    
    def rank_element_importance(self, 
                               data: pd.DataFrame,
                               target_element: str,
                               method: str = "correlation") -> pd.Series:
        """
        排序元素重要性
        
        Args:
            data: 输入数据
            target_element: 目标元素
            method: 重要性计算方法
        
        Returns:
            pd.Series: 元素重要性排序
        """
```

### GeochemProcessor

地球化学数据处理工具。

```python
class GeochemProcessor:
    """地球化学数据处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置参数
        """
    
    def impute_censored_data(self, 
                            data: pd.DataFrame,
                            detection_limits: Dict[str, float],
                            method: str = "rosner") -> pd.DataFrame:
        """
        插补删失数据
        
        Args:
            data: 输入数据
            detection_limits: 检测限字典
            method: 插补方法
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
    
    def transform_clr(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        中心对数比变换
        
        Args:
            data: 输入数据
        
        Returns:
            pd.DataFrame: CLR变换后的数据
        """
    
    def detect_outliers(self, 
                        data: pd.DataFrame,
                        method: str = "iqr") -> pd.DataFrame:
        """
        检测异常值
        
        Args:
            data: 输入数据
            method: 检测方法
        
        Returns:
            pd.DataFrame: 异常值标记
        """
```

### FractalAnomalyFilter

分形异常过滤器。

```python
class FractalAnomalyFilter:
    """分形异常过滤器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化分形过滤器
        
        Args:
            config: 配置参数
        """
    
    def plot_ca_loglog(self, 
                      data: np.ndarray,
                      bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        绘制C-A双对数图
        
        Args:
            data: 输入数据
            bins: 分箱数量
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 面积和浓度数组
        """
    
    def calculate_threshold_interactive(self, 
                                      data: np.ndarray,
                                      method: str = "knee") -> float:
        """
        交互式计算阈值
        
        Args:
            data: 输入数据
            method: 阈值计算方法
        
        Returns:
            float: 异常阈值
        """
    
    def filter_anomalies(self, 
                        data: pd.DataFrame,
                        element: str,
                        threshold: float) -> pd.DataFrame:
        """
        过滤异常值
        
        Args:
            data: 输入数据
            element: 元素名称
            threshold: 阈值
        
        Returns:
            pd.DataFrame: 异常点数据
        """
```

### WeightsOfEvidenceCalculator

证据权重计算器。

```python
class WeightsOfEvidenceCalculator:
    """证据权重计算器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化权重计算器
        
        Args:
            config: 配置参数
        """
    
    def calculate_studentized_contrast(self, 
                                      w_plus: np.ndarray,
                                      w_minus: np.ndarray,
                                      s2_w_plus: np.ndarray,
                                      s2_w_minus: np.ndarray) -> np.ndarray:
        """
        计算学生化对比度
        
        Args:
            w_plus: 正权重
            w_minus: 负权重
            s2_w_plus: 正权重方差
            s2_w_minus: 负权重方差
        
        Returns:
            np.ndarray: 学生化对比度
        """
    
    def calculate_weights(self, 
                         evidence: np.ndarray,
                         target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算证据权重
        
        Args:
            evidence: 证据数据
            target: 目标变量
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: W+, W-, Contrast
        """
    
    def validate_significance(self, 
                             contrast: np.ndarray,
                             studentized_contrast: np.ndarray,
                             alpha: float = 0.05) -> np.ndarray:
        """
        验证统计显著性
        
        Args:
            contrast: 对比度
            studentized_contrast: 学生化对比度
            alpha: 显著性水平
        
        Returns:
            np.ndarray: 显著性标记
        """
```

## ⚙️ 配置模块

### ConfigManager

配置管理器。

```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
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
    
    def save(self, filepath: str) -> None:
        """
        保存配置到文件
        
        Args:
            filepath: 文件路径
        """
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        批量更新配置
        
        Args:
            updates: 更新字典
        """
    
    def get_detection_limits(self) -> Dict[str, float]:
        """
        获取检测限配置
        
        Returns:
            Dict[str, float]: 检测限字典
        """
```

## 🛠️ 实用工具

### 日志工具

```python
def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logging.Logger: 日志记录器
    """
```

### 数据验证工具

```python
def validate_data(data: pd.DataFrame, 
                 required_columns: List[str],
                 check_geometry: bool = False) -> bool:
    """
    验证数据格式
    
    Args:
        data: 输入数据
        required_columns: 必需列
        check_geometry: 是否检查几何信息
    
    Returns:
        bool: 验证结果
    """

def validate_geochemical_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    验证地球化学数据
    
    Args:
        data: 输入数据
    
    Returns:
        Dict[str, Any]: 验证结果
    """
```

### 文件操作工具

```python
def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        filepath: 文件路径
        **kwargs: 额外参数
    
    Returns:
        pd.DataFrame: 加载的数据
    """

def save_results(results: Dict[str, Any], 
                filepath: str,
                format: str = "json") -> None:
    """
    保存结果
    
    Args:
        results: 结果数据
        filepath: 文件路径
        format: 保存格式
    """

def create_output_directory(base_dir: str, 
                           analysis_name: str) -> str:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录
        analysis_name: 分析名称
    
    Returns:
        str: 输出目录路径
    """
```

## 💻 CLI接口

### 主命令

```bash
gold-seeker [OPTIONS] COMMAND [ARGS]...
```

### 子命令

#### analyze

```bash
gold-seeker analyze [OPTIONS] INPUT_FILE TARGET_ELEMENT
```

选项：
- `--config PATH`: 配置文件路径
- `--output PATH`: 输出目录
- `--method TEXT`: 分析方法
- `--elements TEXT`: 指定元素列表
- `--parallel`: 启用并行处理
- `--n-jobs INTEGER`: 并行作业数
- `--verbose`: 详细输出

#### workflow

```bash
gold-seeker workflow [OPTIONS] INPUT_FILE TARGET_ELEMENT
```

选项：
- `--config PATH`: 配置文件路径
- `--workflow-config PATH`: 工作流配置文件
- `--output PATH`: 输出目录
- `--save-intermediate`: 保存中间结果

#### validate

```bash
gold-seeker validate [OPTIONS]
```

选项：
- `--config PATH`: 配置文件路径
- `--data PATH`: 数据文件路径
- `--check-all`: 检查所有组件

#### info

```bash
gold-seeker info [OPTIONS]
```

选项：
- `--version`: 显示版本信息
- `--system`: 显示系统信息
- `--dependencies`: 显示依赖信息

#### example

```bash
gold-seeker example [OPTIONS]
```

选项：
- `--dataset TEXT`: 数据集名称
- `--output PATH`: 输出目录
- `--run`: 运行示例

## 📝 类型定义

### 数据类型

```python
# 基础数据类型
DataFrame = pd.DataFrame
GeoDataFrame = gpd.GeoDataFrame
NDArray = np.ndarray

# 配置类型
ConfigDict = Dict[str, Any]
AnalysisConfig = Dict[str, Any]

# 结果类型
AnalysisResult = Dict[str, Any]
WorkflowResult = Dict[str, Any]
ValidationResult = Dict[str, Any]

# 地理类型
Geometry = shapely.geometry.base.BaseGeometry
CRS = pyproj.CRS

# 时间类型
Timestamp = datetime
TimeDelta = timedelta
```

### 枚举类型

```python
class ModelType(Enum):
    """模型类型枚举"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    WEIGHTS_OF_EVIDENCE = "weights_of_evidence"

class AnalysisMethod(Enum):
    """分析方法枚举"""
    R_MODE_CLUSTERING = "r_mode_clustering"
    PCA_ANALYSIS = "pca_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    MUTUAL_INFORMATION = "mutual_information"

class AnomalyMethod(Enum):
    """异常检测方法枚举"""
    C_A_FRACTAL = "c_a_fractal"
    CONCENTRATION_AREA = "concentration_area"
    STATISTICAL_THRESHOLD = "statistical_threshold"
    MACHINE_LEARNING = "machine_learning"
```

### 数据类

```python
@dataclass
class Task:
    """任务数据类"""
    id: str
    description: str
    priority: int
    status: str
    created_at: datetime
    updated_at: datetime

@dataclass
class WorkflowPlan:
    """工作流计划数据类"""
    tasks: List[Task]
    dependencies: Dict[str, List[str]]
    estimated_duration: timedelta
    resources: Dict[str, Any]

@dataclass
class KnowledgeItem:
    """知识项数据类"""
    id: str
    title: str
    content: str
    source: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class EvidenceLayer:
    """证据层数据类"""
    name: str
    data: GeoDataFrame
    weight: float
    confidence: float
    metadata: Dict[str, Any]
```

---

本API参考文档提供了Gold-Seeker平台的完整接口说明。如需更多详细信息，请参考源代码中的文档字符串和类型注解。