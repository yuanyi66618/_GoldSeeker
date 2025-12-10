"""
Gold-Seeker: AI Mineral Prediction System
Streamlit Frontend Application

基于Streamlit的地球化学找矿预测交互式界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import json
import io
import base64
from pathlib import Path
import sys
import warnings

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 设置seaborn中文字体
sns.set_style('whitegrid')
sns.set_palette('husl')

# 设置Plotly中文字体
import plotly.io as pio
pio.templates.default = "plotly_white"
# 设置中文字体
font_config = {
    'family': 'Microsoft YaHei, SimHei, FangSong, SimSun, Arial',
    'size': 12,
    'color': '#333333'
}
pio.templates["custom"] = {
    'layout': {
        'font': font_config,
        'title': {
            'font': {
                'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                'size': 16
            }
        },
        'xaxis': {
            'title': {
                'font': {
                    'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                    'size': 14
                }
            },
            'tickfont': {
                'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                'size': 12
            }
        },
        'yaxis': {
            'title': {
                'font': {
                    'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                    'size': 14
                }
            },
            'tickfont': {
                'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                'size': 12
            }
        }
    }
}

# 设置页面配置
st.set_page_config(
    page_title="Gold-Seeker: AI Mineral Prediction System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
def set_custom_style():
    """设置自定义样式"""
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .stSidebar {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2c3e50;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #34495e;
        color: white;
    }
    .plot-container {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .chat-message {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .agent-message {
        background-color: rgba(52, 152, 219, 0.2);
        border-left: 4px solid #3498db;
    }
    .user-message {
        background-color: rgba(46, 204, 113, 0.2);
        border-left: 4px solid #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化session state
def init_session_state():
    """初始化session state"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'selected_elements' not in st.session_state:
        st.session_state.selected_elements = ['Au', 'As', 'Sb', 'Hg']
    if 'target_mineral' not in st.session_state:
        st.session_state.target_mineral = 'Au'

# 生成模拟数据
def generate_mock_data(n_samples=200):
    """生成模拟地球化学数据"""
    np.random.seed(42)
    
    data = {
        'X': np.random.uniform(0, 1000, n_samples),
        'Y': np.random.uniform(0, 1000, n_samples),
        'Au': np.random.lognormal(0, 1, n_samples),
        'As': np.random.lognormal(1, 0.8, n_samples),
        'Sb': np.random.lognormal(0.5, 0.9, n_samples),
        'Hg': np.random.lognormal(-0.5, 1.2, n_samples),
        'Cu': np.random.lognormal(2, 0.7, n_samples),
        'Pb': np.random.lognormal(1.5, 0.6, n_samples),
        'Zn': np.random.lognormal(2.2, 0.5, n_samples),
        'Ag': np.random.lognormal(-0.2, 1.0, n_samples),
    }
    
    # 添加一些低于检测限的值
    detection_limits = {'Au': 0.05, 'As': 0.5, 'Sb': 0.2, 'Hg': 0.01}
    for element, limit in detection_limits.items():
        censored_mask = np.random.random(n_samples) < 0.2
        data[element][censored_mask] = np.random.uniform(0, limit, censored_mask.sum())
    
    # 添加训练点标签
    data['Is_Deposit'] = np.zeros(n_samples, dtype=int)
    deposit_indices = np.random.choice(n_samples, size=20, replace=False)
    for idx in deposit_indices:
        data['Is_Deposit'][idx] = 1
        data['Au'][idx] *= np.random.uniform(5, 20)
        data['As'][idx] *= np.random.uniform(3, 10)
        data['Sb'][idx] *= np.random.uniform(2, 8)
    
    return pd.DataFrame(data)

# 生成相关性热力图
def create_correlation_heatmap(data, elements):
    """创建相关性热力图"""
    corr_matrix = data[elements].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax, cbar_kws={'label': '相关系数'})
    ax.set_title('元素相关性热力图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# 生成R型聚类树状图
def create_dendrogram(data, elements):
    """创建R型聚类树状图"""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    
    # 计算相关性距离
    corr_matrix = data[elements].corr()
    distance_matrix = 1 - np.abs(corr_matrix)
    condensed_distances = pdist(distance_matrix.values)
    
    # 层次聚类
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=elements, ax=ax, 
               leaf_rotation=45, leaf_font_size=12)
    ax.set_title('R型聚类树状图', fontsize=16, fontweight='bold')
    ax.set_xlabel('元素', fontsize=12)
    ax.set_ylabel('距离', fontsize=12)
    plt.tight_layout()
    
    return fig

# 生成PCA载荷图
def create_pca_loadings_plot(data, elements):
    """创建PCA载荷图"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[elements])
    
    # PCA分析
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    
    # 创建载荷图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制载荷向量
    for i, element in enumerate(elements):
        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, 
                element, fontsize=12, ha='center', va='center')
    
    # 添加参考圆
    circle = Circle((0, 0), 1, fill=False, color='blue', linestyle='--')
    ax.add_patch(circle)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=12)
    ax.set_title('PCA载荷图', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    return fig

# 生成C-A分形图
def create_ca_fractal_plot(data, element):
    """创建C-A分形图"""
    # 模拟C-A分形分析
    concentrations = np.sort(data[element].values)
    areas = np.arange(1, len(concentrations) + 1)
    
    # 对数变换
    log_conc = np.log10(concentrations[concentrations > 0])
    log_area = np.log10(areas[concentrations > 0])
    
    # 模拟拐点
    threshold_idx = int(len(log_conc) * 0.8)
    threshold = concentrations[threshold_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点图
    ax.scatter(log_conc, log_area, alpha=0.6, s=30, c='blue', label='数据点')
    
    # 拟合背景线
    bg_mask = log_conc < np.log10(threshold)
    if bg_mask.sum() > 1:
        bg_fit = np.polyfit(log_conc[bg_mask], log_area[bg_mask], 1)
        bg_line = np.poly1d(bg_fit)
        ax.plot(log_conc[bg_mask], bg_line(log_conc[bg_mask]), 
                'r--', linewidth=2, label='背景拟合')
    
    # 拟合异常线
    anom_mask = log_conc >= np.log10(threshold)
    if anom_mask.sum() > 1:
        anom_fit = np.polyfit(log_conc[anom_mask], log_area[anom_mask], 1)
        anom_line = np.poly1d(anom_fit)
        ax.plot(log_conc[anom_mask], anom_line(log_conc[anom_mask]), 
                'g--', linewidth=2, label='异常拟合')
    
    # 标记拐点
    ax.axvline(x=np.log10(threshold), color='red', linestyle=':', 
               linewidth=2, label=f'阈值: {threshold:.3f}')
    
    ax.set_xlabel('log(浓度)', fontsize=12)
    ax.set_ylabel('log(面积)', fontsize=12)
    ax.set_title(f'{element} C-A分形分析', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, threshold

# 创建交互式地图
def create_interactive_map(data, element, threshold=None):
    """创建交互式地图"""
    # 计算中心点
    center_lat = data['Y'].mean()
    center_lon = data['X'].mean()
    
    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # 添加采样点
    for idx, row in data.iterrows():
        color = 'red' if row.get('Is_Deposit', 0) == 1 else 'blue'
        size = 8 if row.get('Is_Deposit', 0) == 1 else 5
        
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=size,
            popup=f"点位 {idx}<br>{element}: {row[element]:.3f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # 如果有阈值，添加异常区域
    if threshold is not None:
        anomaly_points = data[data[element] > threshold]
        if len(anomaly_points) > 0:
            # 创建异常区域的凸包
            from scipy.spatial import ConvexHull
            points = anomaly_points[['Y', 'X']].values
            
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # 创建多边形
                    folium.Polygon(
                        locations=[[p[0], p[1]] for p in hull_points],
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.2,
                        popup='异常区域'
                    ).add_to(m)
                except:
                    pass
    
    return m

# 模拟Agent响应
def mock_agent_response(user_input):
    """模拟Agent响应"""
    responses = {
        "相关性": "我正在分析元素之间的相关性。根据计算结果，Au与As的相关系数为0.75，显示出强烈的正相关性，这是金矿成矿的重要地球化学指标。",
        "异常": "我已经完成了智能异常检测分析，识别出Au的异常阈值为1.2 ppb，共有15个样品被归类为异常，这些区域值得进一步勘探。",
        "聚类": "基于机器学习的聚类分析显示，Au、As、Sb、Hg形成一个紧密的元素组合，这是典型的金矿化元素组合特征。",
        "预测": "通过融合地质知识图谱与大模型的智能预测系统，研究区的成矿潜力评分为0.75，属于高潜力区域。",
        "勘探": "根据智能体分析，建议重点关注构造断裂带附近的异常区域，这些区域具有较好的成矿地质条件。",
        "模型": "本平台采用多模态大模型，融合了地质学、地球化学、遥感等多源数据，提供精准的金矿预测服务。"
    }
    
    for key, response in responses.items():
        if key in user_input:
            return response
    
    return "我是金矿智能预测专家，正在分析您的请求。我可以为您提供成矿预测、异常识别、勘探建议等专业服务。"

# 侧边栏配置
def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>⛏️ Gold-Seeker</h1>
        <p style='font-size: 14px; opacity: 0.8;'>金矿智能预测智能体平台</p>
        <p style='font-size: 12px; opacity: 0.6;'>融合领域知识与大模型技术</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 数据上传
    st.sidebar.markdown("### 📁 数据上传")
    uploaded_file = st.sidebar.file_uploader(
        "选择CSV或GeoJSON文件",
        type=['csv', 'geojson'],
        help="上传地球化学数据文件"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                # 简单的GeoJSON处理
                import geopandas as gpd
                gdf = gpd.read_file(uploaded_file)
                data = pd.DataFrame(gdf.drop(columns='geometry'))
            
            st.session_state.data = data
            st.sidebar.success(f"✅ 成功加载数据: {data.shape}")
        except Exception as e:
            st.sidebar.error(f"❌ 加载失败: {str(e)}")
    
    # 使用示例数据
    if st.sidebar.button("🎲 使用示例数据"):
        st.session_state.data = generate_mock_data()
        st.sidebar.success("✅ 已加载示例数据")
    
    # 参数设置
    st.sidebar.markdown("### ⚙️ 参数设置")
    
    # 选择目标矿种
    target_mineral = st.sidebar.selectbox(
        "目标矿种",
        ['Au', 'Ag', 'Cu', 'Pb', 'Zn'],
        index=0,
        help="选择主要找矿目标元素"
    )
    st.session_state.target_mineral = target_mineral
    
    # 选择分析元素
    if st.session_state.data is not None:
        available_elements = [col for col in st.session_state.data.columns 
                           if col not in ['X', 'Y', 'Is_Deposit']]
        
        selected_elements = st.sidebar.multiselect(
            "分析元素",
            available_elements,
            default=['Au', 'As', 'Sb', 'Hg'] if all(e in available_elements for e in ['Au', 'As', 'Sb', 'Hg']) else available_elements[:4],
            help="选择要分析的元素"
        )
        st.session_state.selected_elements = selected_elements
    
    # 初始化Agent
    st.sidebar.markdown("### 🤖 初始化智能体")
    if st.sidebar.button("🚀 Initialize Agent", type="primary"):
        if st.session_state.data is not None:
            # TODO: 替换为真实的SpatialAnalystAgent初始化
            st.session_state.agent = "Mock Agent"
            st.sidebar.success("✅ Agent已初始化")
        else:
            st.sidebar.error("❌ 请先上传数据")

# Agent聊天界面
def render_agent_chat():
    """渲染Agent聊天界面"""
    st.markdown("### 🤖 金矿智能预测对话")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p>🤖 <strong>智能体介绍：</strong>我是融合地质领域知识与先进大模型技术的金矿智能预测专家，
        能够为您提供专业的金矿勘探建议、数据分析和成矿预测服务。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示聊天历史
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 用户:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>🤖 Agent:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # 用户输入
    user_input = st.text_input("💬 输入您的问题:", key="user_input")
    
    if st.button("📤 发送") and user_input:
        # 添加用户消息
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # 模拟Agent响应
        # TODO: 替换为真实的SpatialAnalystAgent调用
        agent_response = mock_agent_response(user_input)
        
        # 添加Agent响应
        st.session_state.chat_history.append({
            'role': 'agent',
            'content': agent_response
        })
        
        # 清空输入框
        st.session_state.user_input = ""
        
        # 重新运行以显示新消息
        st.rerun()
    
    # 清空聊天历史
    if st.button("🗑️ 清空聊天历史"):
        st.session_state.chat_history = []
        st.rerun()

# 数据分析界面
def render_data_analysis():
    """渲染数据分析界面"""
    st.markdown("### 📊 数据预览")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        elements = st.session_state.selected_elements
        
        # 数据概览
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("样本数量", len(data))
        with col2:
            st.metric("分析元素", len(elements))
        with col3:
            st.metric("目标矿种", st.session_state.target_mineral)
        
        # 数据表格
        st.markdown("#### 📋 数据表格")
        st.dataframe(data.head(10))
        
        # 统计信息
        st.markdown("#### 📈 统计信息")
        if elements:
            stats_data = data[elements].describe()
            st.dataframe(stats_data)
        
        # 可视化区域
        st.markdown("#### 📊 可视化分析")
        
        if len(elements) >= 2:
            # 相关性热力图
            with st.expander("🔥 相关性热力图", expanded=True):
                fig = create_correlation_heatmap(data, elements)
                st.pyplot(fig)
                plt.close()
            
            # R型聚类树状图
            with st.expander("🌳 R型聚类树状图", expanded=True):
                fig = create_dendrogram(data, elements)
                st.pyplot(fig)
                plt.close()
            
            # PCA载荷图
            with st.expander("🎯 PCA载荷图", expanded=True):
                fig = create_pca_loadings_plot(data, elements)
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("⚠️ 请至少选择2个元素进行分析")
    else:
        st.warning("⚠️ 请先上传数据")

# 空间分析界面
def render_spatial_analysis():
    """渲染空间分析界面"""
    st.markdown("### 🗺️ 空间分析")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        target_element = st.session_state.target_mineral
        
        # 选择分析元素
        analysis_element = st.selectbox(
            "选择分析元素",
            st.session_state.selected_elements,
            index=0 if st.session_state.selected_elements else 0
        )
        
        # C-A分形分析
        st.markdown("#### 📈 C-A分形分析")
        
        with st.expander("🔍 C-A分形图", expanded=True):
            fig, threshold = create_ca_fractal_plot(data, analysis_element)
            st.pyplot(fig)
            plt.close()
            
            if threshold:
                st.info(f"📍 识别的异常阈值: {threshold:.3f}")
        
        # 交互式地图
        st.markdown("#### 🗺️ 交互式地图")
        
        with st.expander("🌍 空间分布图", expanded=True):
            # 创建地图
            m = create_interactive_map(data, analysis_element, threshold)
            
            # 显示地图
            map_data = st_folium(m, width=700, height=500)
            
            # 显示地图统计
            if map_data['last_clicked']:
                st.info(f"📍 最后点击位置: {map_data['last_clicked']}")
        
        # 异常统计
        if threshold:
            anomaly_count = (data[analysis_element] > threshold).sum()
            st.markdown("#### 📊 异常统计")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("异常样品数", anomaly_count)
            with col2:
                st.metric("异常率", f"{anomaly_count/len(data)*100:.1f}%")
            with col3:
                st.metric("阈值", f"{threshold:.3f}")
    else:
        st.warning("⚠️ 请先上传数据")

# 主函数
def main():
    """主函数"""
    # 设置自定义样式
    set_custom_style()
    
    # 初始化session state
    init_session_state()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主界面标题
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1>⛏️ Gold-Seeker: AI Mineral Prediction System</h1>
        <p style='font-size: 18px; opacity: 0.9;'>融合领域知识与大模型的金矿智能预测智能体平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🤖 Agent Chat", "📊 Data & R-mode Analysis", "🗺️ Spatial & Anomaly"])
    
    with tab1:
        render_agent_chat()
    
    with tab2:
        render_data_analysis()
    
    with tab3:
        render_spatial_analysis()
    
    # 页脚
    st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 50px; border-top: 1px solid rgba(255,255,255,0.2);'>
        <p>© 2025 Gold-Seeker Development Team | 融合领域知识与大模型的金矿智能预测智能体平台</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings('ignore')
    
    # 运行应用
    main()