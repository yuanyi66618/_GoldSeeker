"""
Gold-Seeker Streamlit App Launcher
启动Streamlit应用的便捷脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """主函数"""
    # 设置环境变量以跳过邮箱输入
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # 检查是否安装了streamlit
    try:
        import streamlit
        print(f"✅ Streamlit已安装: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit未安装，请先安装依赖:")
        print("pip install -r requirements-streamlit.txt")
        return 1
    
    # 设置项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"🚀 启动Gold-Seeker应用...")
    print(f"📁 项目目录: {project_root}")
    print(f"🌐 应用地址: http://localhost:8508")
    print("=" * 50)
    
    # 启动streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8508",
            "--server.address", "0.0.0.0",  # 允许外部访问
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#FF6B6B",
            "--theme.backgroundColor", "#1E3C72",
            "--theme.secondaryBackgroundColor", "#2A5298",
            "--theme.textColor", "#FFFFFF",
            "--theme.font", "sans serif"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
        return 0

if __name__ == "__main__":
    sys.exit(main())