"""
检查系统中文字体
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 获取所有可用字体
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = []

# 查找中文字体
for font in fonts:
    if any(keyword in font.lower() for keyword in ['microsoft', 'yahei', 'simhei', 'simsun', 'fangsong']):
        chinese_fonts.append(font)

print("系统中可用的中文字体：")
for font in sorted(set(chinese_fonts)):
    print(f"  - {font}")

print(f"\n总共找到 {len(set(chinese_fonts))} 个中文字体")

# 测试字体设置
print("\n测试字体设置...")

# 尝试设置字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建简单测试
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, '中文字体测试：金矿智能预测', 
            fontsize=16, ha='center', va='center', transform=ax.transAxes)
    ax.set_title('字体测试', fontsize=14)
    ax.axis('off')
    
    plt.savefig('font_check.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ 字体设置成功！生成了 font_check.png 文件")
    
except Exception as e:
    print(f"❌ 字体设置失败：{e}")
    
    # 尝试使用备用字体
    print("尝试使用备用字体...")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, '中文字体测试：金矿智能预测', 
                fontsize=16, ha='center', va='center', transform=ax.transAxes)
        ax.set_title('字体测试', fontsize=14)
        ax.axis('off')
        plt.savefig('font_check_backup.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 备用字体设置成功！")
    except Exception as e2:
        print(f"❌ 备用字体设置也失败：{e2}")