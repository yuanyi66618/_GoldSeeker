"""
Gold-Seeker Frontend Test Script
测试前端应用的基本功能
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """测试必要的库导入"""
    print("🔍 测试库导入...")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit: {st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib: {plt.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib导入失败: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn: {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn导入失败: {e}")
        return False
    
    try:
        import plotly.express as px
        print(f"✅ Plotly: {px.__version__}")
    except ImportError as e:
        print(f"❌ Plotly导入失败: {e}")
        return False
    
    try:
        import folium
        print(f"✅ Folium: {folium.__version__}")
    except ImportError as e:
        print(f"❌ Folium导入失败: {e}")
        return False
    
    try:
        import streamlit_folium
        print(f"✅ Streamlit-Folium: {streamlit_folium.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit-Folium导入失败: {e}")
        return False
    
    try:
        import geopandas as gpd
        print(f"✅ GeoPandas: {gpd.__version__}")
    except ImportError as e:
        print(f"❌ GeoPandas导入失败: {e}")
        return False
    
    return True

def test_app_import():
    """测试应用文件导入"""
    print("\n🔍 测试应用文件导入...")
    
    try:
        # 添加项目路径
        sys.path.insert(0, str(Path(__file__).parent))
        
        # 尝试导入app模块（不运行main函数）
        import app
        print("✅ app.py 导入成功")
        
        # 检查主要函数是否存在
        if hasattr(app, 'main'):
            print("✅ main函数存在")
        else:
            print("❌ main函数不存在")
            return False
        
        if hasattr(app, 'generate_mock_data'):
            print("✅ generate_mock_data函数存在")
        else:
            print("❌ generate_mock_data函数不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ app.py导入失败: {e}")
        traceback.print_exc()
        return False

def test_mock_data():
    """测试模拟数据生成"""
    print("\n🔍 测试模拟数据生成...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import app
        
        # 生成测试数据
        data = app.generate_mock_data(n_samples=50)
        
        print(f"✅ 数据生成成功: {data.shape}")
        print(f"✅ 列名: {list(data.columns)}")
        
        # 检查基本统计
        if 'Au' in data.columns:
            print(f"✅ Au范围: {data['Au'].min():.3f} - {data['Au'].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟数据生成失败: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n🔍 测试可视化功能...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import app
        
        # 生成测试数据
        data = app.generate_mock_data(n_samples=50)
        elements = ['Au', 'As', 'Sb', 'Hg']
        
        # 测试相关性热力图
        fig1 = app.create_correlation_heatmap(data, elements)
        print("✅ 相关性热力图生成成功")
        
        # 测试树状图
        fig2 = app.create_dendrogram(data, elements)
        print("✅ 树状图生成成功")
        
        # 测试PCA载荷图
        fig3 = app.create_pca_loadings_plot(data, elements)
        print("✅ PCA载荷图生成成功")
        
        # 测试C-A分形图
        fig4, threshold = app.create_ca_fractal_plot(data, 'Au')
        print(f"✅ C-A分形图生成成功，阈值: {threshold:.3f}")
        
        # 测试地图
        m = app.create_interactive_map(data, 'Au', threshold)
        print("✅ 交互式地图生成成功")
        
        # 清理图形
        import matplotlib.pyplot as plt
        plt.close('all')
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 Gold-Seeker 前端测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("库导入测试", test_imports),
        ("应用导入测试", test_app_import),
        ("模拟数据测试", test_mock_data),
        ("可视化测试", test_visualization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！前端应用准备就绪。")
        print("\n🚀 运行以下命令启动应用:")
        print("  python run_app.py")
        print("  或")
        print("  streamlit run app.py")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查依赖安装。")
        print("\n📦 安装依赖:")
        print("  pip install -r requirements-streamlit.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())