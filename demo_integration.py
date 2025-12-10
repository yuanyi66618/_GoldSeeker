"""
Gold-Seeker Agent Integration Demo
演示如何将真实的SpatialAnalystAgent集成到Streamlit应用中
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def demo_real_agent_integration():
    """演示真实Agent集成"""
    st.markdown("## 🤖 真实Agent集成演示")
    
    st.markdown("""
    ### 📋 集成步骤
    
    1. **导入Agent模块**
    ```python
    from agents.spatial_analyst import SpatialAnalystAgent
    from langchain_openai import ChatOpenAI
    ```
    
    2. **初始化Agent**
    ```python
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    detection_limits = {'Au': 0.05, 'As': 0.5, 'Sb': 0.2, 'Hg': 0.01}
    agent = SpatialAnalystAgent(llm, detection_limits)
    ```
    
    3. **调用Agent分析**
    ```python
    result = agent.analyze_geochemical_data(
        data=data,
        elements=['Au', 'As', 'Sb', 'Hg'],
        training_points=training_points
    )
    ```
    
    4. **生成报告**
    ```python
    report = agent.generate_analysis_report(result)
    ```
    """)
    
    st.markdown("### 🔧 代码示例")
    
    # 显示示例代码
    example_code = '''
# 在app.py中替换mock函数
def real_agent_response(user_input, agent, data, elements):
    """使用真实Agent响应"""
    try:
        # 根据用户输入选择分析方法
        if "相关性" in user_input:
            # 调用Agent进行相关性分析
            result = agent.analyze_correlations(data, elements)
            return f"相关性分析完成：{result}"
        
        elif "异常" in user_input:
            # 调用Agent进行异常检测
            result = agent.detect_anomalies(data, elements)
            return f"异常检测完成：{result}"
        
        elif "聚类" in user_input:
            # 调用Agent进行聚类分析
            result = agent.perform_clustering(data, elements)
            return f"聚类分析完成：{result}"
        
        else:
            # 通用分析
            result = agent.analyze_geochemical_data(data, elements)
            return f"分析完成：{result}"
    
    except Exception as e:
        return f"分析失败：{str(e)}"

# 在render_agent_chat函数中使用
def render_agent_chat():
    """渲染Agent聊天界面（真实版本）"""
    # ... 现有代码 ...
    
    if st.button("📤 发送") and user_input:
        # 添加用户消息
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # 使用真实Agent响应
        if st.session_state.agent and st.session_state.data:
            agent_response = real_agent_response(
                user_input,
                st.session_state.agent,
                st.session_state.data,
                st.session_state.selected_elements
            )
        else:
            agent_response = "请先初始化Agent并加载数据"
        
        # 添加Agent响应
        st.session_state.chat_history.append({
            'role': 'agent',
            'content': agent_response
        })
        
        st.rerun()
'''
    
    st.code(example_code, language='python')
    
    st.markdown("### 🎯 集成要点")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ 推荐做法**
        - 添加错误处理和重试机制
        - 使用进度条显示长时间运行的任务
        - 缓存分析结果避免重复计算
        - 提供详细的状态反馈
        """)
    
    with col2:
        st.markdown("""
        **⚠️ 注意事项**
        - API调用可能需要时间
        - 需要有效的API密钥
        - 大数据集可能超时
        - 考虑异步处理
        """)

def demo_error_handling():
    """演示错误处理"""
    st.markdown("## 🛡️ 错误处理示例")
    
    error_handling_code = '''
import time
from streamlit.runtime.scriptrunner import RerunData, RerunException

def safe_agent_call(agent, data, elements, max_retries=3):
    """安全的Agent调用"""
    for attempt in range(max_retries):
        try:
            # 显示进度
            with st.spinner(f"正在分析... (尝试 {attempt + 1}/{max_retries})"):
                result = agent.analyze_geochemical_data(data, elements)
                return result
        
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"分析失败: {str(e)}")
                return None
            else:
                st.warning(f"分析出错，正在重试... ({str(e)})")
                time.sleep(2 ** attempt)  # 指数退避
    
    return None

# 在UI中使用
if st.button("🚀 开始分析"):
    if st.session_state.agent and st.session_state.data:
        result = safe_agent_call(
            st.session_state.agent,
            st.session_state.data,
            st.session_state.selected_elements
        )
        
        if result:
            st.success("✅ 分析完成")
            # 处理结果
    else:
        st.error("❌ 请先初始化Agent并加载数据")
'''
    
    st.code(error_handling_code, language='python')

def demo_caching():
    """演示缓存机制"""
    st.markdown("## 💾 缓存机制示例")
    
    caching_code = '''
import streamlit as st

# 使用Streamlit缓存
@st.cache_data(ttl=3600)  # 缓存1小时
def cached_agent_analysis(data_hash, elements, agent_config):
    """缓存的Agent分析"""
    # 重新创建Agent（因为Agent对象不能缓存）
    agent = create_agent(agent_config)
    
    # 执行分析
    result = agent.analyze_geochemical_data(data, elements)
    
    return result

def get_data_hash(data):
    """计算数据哈希"""
    return hash(pd.util.hash_pandas_object(data).sum())

# 在UI中使用
if st.button("🔍 分析数据"):
    if st.session_state.data is not None:
        # 计算数据哈希
        data_hash = get_data_hash(st.session_state.data)
        
        # 获取Agent配置
        agent_config = {
            'model': 'gpt-4',
            'temperature': 0,
            'detection_limits': st.session_state.detection_limits
        }
        
        # 执行缓存分析
        with st.spinner("正在分析..."):
            result = cached_agent_analysis(
                data_hash,
                st.session_state.selected_elements,
                agent_config
            )
        
        st.success("✅ 分析完成（使用缓存）")
    else:
        st.error("❌ 请先加载数据")
'''
    
    st.code(caching_code, language='python')

def main():
    """主函数"""
    st.set_page_config(
        page_title="Gold-Seeker Agent Integration Demo",
        layout="wide"
    )
    
    st.markdown("# 🤖 Gold-Seeker Agent集成演示")
    st.markdown("本页面演示如何将真实的SpatialAnalystAgent集成到Streamlit应用中。")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📋 集成步骤", "🛡️ 错误处理", "💾 缓存机制"])
    
    with tab1:
        demo_real_agent_integration()
    
    with tab2:
        demo_error_handling()
    
    with tab3:
        demo_caching()
    
    # 实际集成提示
    st.markdown("---")
    st.markdown("### 🚀 开始集成")
    
    st.markdown("""
    1. **确保后端模块正常工作**
       ```bash
       python -m pytest tests/test_spatial_analyst.py
       ```
    
    2. **配置API密钥**
       ```bash
       export OPENAI_API_KEY="your-api-key"
       ```
    
    3. **修改app.py**
       - 替换`mock_agent_response`函数
       - 添加真实Agent初始化
       - 集成错误处理和缓存
    
    4. **测试集成**
       ```bash
       python test_frontend.py
       streamlit run app.py
       ```
    """)
    
    if st.button("🧪 运行集成测试", type="primary"):
        st.info("请参考上述代码示例，将真实Agent集成到app.py中。")

if __name__ == "__main__":
    main()