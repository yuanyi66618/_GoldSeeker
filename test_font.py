"""
测试中文字体设置
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 设置seaborn样式
sns.set_style('whitegrid')

# 创建测试数据
np.random.seed(42)
data = pd.DataFrame({
    '金': np.random.normal(10, 2, 100),
    '银': np.random.normal(8, 1.5, 100),
    '铜': np.random.normal(15, 3, 100)
})

# 测试图表
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 相关性热力图
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=axes[0,0], cbar_kws={'label': '相关系数'})
axes[0,0].set_title('元素相关性热力图', fontsize=14, fontweight='bold')

# 2. 散点图
axes[0,1].scatter(data['金'], data['银'], alpha=0.6)
axes[0,1].set_xlabel('金含量', fontsize=12)
axes[0,1].set_ylabel('银含量', fontsize=12)
axes[0,1].set_title('金-银散点图', fontsize=14, fontweight='bold')

# 3. 箱线图
data.boxplot(ax=axes[1,0])
axes[1,0].set_title('元素含量箱线图', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('含量', fontsize=12)

# 4. 直方图
axes[1,1].hist(data['金'], bins=20, alpha=0.7, color='gold')
axes[1,1].set_xlabel('金含量', fontsize=12)
axes[1,1].set_ylabel('频数', fontsize=12)
axes[1,1].set_title('金含量分布直方图', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('font_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("字体测试完成！请检查生成的font_test.png文件中的中文字体显示是否正常。")
print("如果中文字体显示正常，说明字体设置成功。")
print("如果显示为方框，说明字体设置有问题。")