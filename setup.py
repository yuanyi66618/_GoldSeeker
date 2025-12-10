"""
Gold-Seeker: 地球化学找矿预测智能平台
基于Carranza (2009) 理论的智能地球化学找矿预测平台
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# 版本信息
VERSION = "1.0.0"

setup(
    name="gold-seeker",
    version=VERSION,
    author="Gold-Seeker Development Team",
    author_email="your-email@example.com",
    description="地球化学找矿预测智能平台",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Gold-Seeker",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/Gold-Seeker/issues",
        "Documentation": "https://gold-seeker.readthedocs.io/",
        "Source Code": "https://github.com/your-username/Gold-Seeker",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
            "torch>=1.12.0",
            "tensorflow>=2.10.0",
        ],
        "web": [
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
            "streamlit>=1.0.0",
        ],
        "database": [
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
            "pymongo>=3.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gold-seeker=agents.cli:main",
            "gold-seeker-analyst=agents.spatial_analyst:main",
            "gold-seeker-coordinator=agents.coordinator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agents": [
            "config/*.yaml",
            "templates/*.txt",
            "data/examples/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "geochemistry",
        "mineral-exploration",
        "geological-survey",
        "machine-learning",
        "spatial-analysis",
        "fractal-analysis",
        "weights-of-evidence",
        "langchain",
        "multi-agent",
        "gold-prospecting",
        "caranza-method",
    ],
    license="MIT",
    platforms=["any"],
)