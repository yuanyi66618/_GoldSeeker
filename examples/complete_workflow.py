"""
Gold-Seeker 完整工作流示例

演示从原始地球化学数据到成矿预测的完整流程：
1. 数据加载与预处理
2. 元素组合识别
3. 分形异常滤波
4. 证据权计算
5. 结果可视化与报告生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.tools.geochem import (
    GeochemSelector,
    GeochemProcessor,
    FractalAnomalyFilter,
    WeightsOfEvidenceCalculator
)


def generate_synthetic_geochemical_data(n_samples=1000, seed=42):
    """
    生成合成地球化学数据用于演示
    
    模拟卡林型金矿的地球化学特征：
    - Au-As-Sb-Hg 元素组合
    - 部分样品受检测限影响
    - 包含空间相关的异常
    """
    np.random.seed(seed)
    
    # 基础背景数据
    background_data = {
        'Au': np.random.lognormal(-1.0, 0.8, n_samples),
        'As': np.random.lognormal(0.5, 0.7, n_samples),
        'Sb': np.random.lognormal(0.2, 0.9, n_samples),
        'Hg': np.random.lognormal(-0.5, 1.0, n_samples),
        'Cu': np.random.lognormal(1.0, 0.6, n_samples),
        'Pb': np.random.lognormal(0.8, 0.5, n_samples),
        'Zn': np.random.lognormal(1.2, 0.4, n_samples),
        'Ag': np.random.lognormal(-0.8, 0.9, n_samples)
    }
    
    # 添加元素相关性（模拟共生组合）
    # Au-As-Sb-Hg 组合
    gold_association = np.random.lognormal(1.5, 0.5, n_samples // 10)
    association_mask = np.random.choice(n_samples, n_samples // 10, replace=False)
    
    for i in association_mask:
        background_data['Au'][i] += gold_association[np.where(association_mask == i)[0][0]]
        background_data['As'][i] += 0.7 * gold_association[np.where(association_mask == i)[0][0]]
        background_data['Sb'][i] += 0.5 * gold_association[np.where(association_mask == i)[0][0]]
        background_data['Hg'][i] += 0.3 * gold_association[np.where(association_mask == i)[0][0]]
    
    # 添加检测限以下数据
    detection_limits = {'Au': 0.05, 'As': 0.5, 'Sb': 0.2, 'Hg': 0.02}
    
    for element, limit in detection_limits.items():
        n_censored = n_samples // 20  # 5%的数据低于检测限
        censored_indices = np.random.choice(n_samples, n_censored, replace=False)
        background_data[element][censored_indices] = np.random.uniform(0, limit * 0.8, n_censored)
    
    # 创建DataFrame
    df = pd.DataFrame(background_data)
    
    # 添加坐标信息（模拟空间分布）
    df['X'] = np.random.uniform(0, 100, n_samples)
    df['Y'] = np.random.uniform(0, 100, n_samples)
    
    # 创建训练点（矿点标记）
    # 高Au值区域更有可能是矿点
    au_threshold = np.percentile(df['Au'], 90)
    df['Is_Deposit'] = ((df['Au'] > au_threshold) & 
                       (df['As'] > np.percentile(df['As'], 85))).astype(int)
    
    return df, detection_limits


def main():
    """主工作流程"""
    print("=" * 60)
    print("Gold-Seeker 地球化学证据层构建完整工作流")
    print("=" * 60)
    
    # 1. 数据生成
    print("\n1. 生成合成地球化学数据...")
    geochem_data, detection_limits = generate_synthetic_geochemical_data(n_samples=1000)
    print(f"   数据形状: {geochem_data.shape}")
    print(f"   检测限: {detection_limits}")
    print(f"   矿点数量: {geochem_data['Is_Deposit'].sum()}")
    
    # 2. 元素组合识别
    print("\n2. 执行R型聚类分析识别元素组合...")
    selector = GeochemSelector(detection_limits)
    
    # R-mode聚类分析
    elements = ['Au', 'As', 'Sb', 'Hg', 'Cu', 'Pb', 'Zn', 'Ag']
    r_mode_result = selector.perform_r_mode_analysis(
        geochem_data, 
        elements=elements,
        method='ward'
    )
    
    print(f"   识别出 {len(r_mode_result['clusters'])} 个元素组合:")
    for cluster_id, cluster_elements in r_mode_result['clusters'].items():
        print(f"     组合 {cluster_id}: {', '.join(cluster_elements)}")
    
    # PCA分析
    pca_result = selector.analyze_pca_loadings(
        geochem_data,
        elements=elements,
        n_components=3
    )
    
    print(f"   推荐的找矿元素组合: {', '.join(pca_result['recommended_elements'])}")
    print(f"   前3个主成分解释方差: {pca_result['explained_variance_ratio']}")
    print(f"   累积方差解释: {pca_result['cumulative_variance_ratio'][-1]:.3f}")
    
    # 3. 数据预处理
    print("\n3. 数据预处理...")
    processor = GeochemProcessor(detection_limits)
    
    # 处理检测限数据
    processed_data = processor.impute_censored_data(
        geochem_data,
        elements=elements,
        method='substitution'
    )
    print(f"   检测限数据处理完成")
    
    # CLR变换
    clr_data = processor.transform_clr(
        processed_data,
        elements=elements
    )
    print(f"   CLR变换完成，变换后数据形状: {clr_data.shape}")
    
    # 异常值检测
    outlier_result = processor.detect_outliers(
        processed_data,
        elements=['Au', 'As', 'Sb'],
        method='robust'
    )
    print(f"   检测到 {outlier_result['summary']['outlier_samples']} 个异常样品")
    
    # 4. 分形异常滤波
    print("\n4. C-A分形异常滤波...")
    fractal_filter = FractalAnomalyFilter()
    
    # 对Au进行分形分析
    au_concentrations = processed_data['Au'].values
    ca_plot = fractal_filter.plot_ca_loglog(
        au_concentrations,
        element_name='Au'
    )
    
    # 计算分形阈值
    threshold_result = fractal_filter.calculate_threshold_interactive(
        au_concentrations,
        element_name='Au',
        method='knee'
    )
    
    print(f"   Au异常阈值: {threshold_result['threshold']:.4f}")
    print(f"   异常样品数: {threshold_result['n_anomalies']}")
    print(f"   异常比例: {threshold_result['anomaly_percentage']:.1f}%")
    
    # 5. 证据权计算
    print("\n5. 证据权计算...")
    woe_calculator = WeightsOfEvidenceCalculator()
    
    # 准备训练数据
    training_points = geochem_data['Is_Deposit'].values
    
    # 计算Au的证据权
    woe_result = woe_calculator.calculate_studentized_contrast(
        processed_data['Au'],
        training_points,
        evidence_classes=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    print(f"   证据权计算完成，共 {len(woe_result)} 个类别")
    
    # 显示显著的类别
    significant_classes = woe_result[woe_result['Significant']]
    if len(significant_classes) > 0:
        print("   显著类别:")
        for _, row in significant_classes.iterrows():
            print(f"     {row['Class_Range']}: C={row['Contrast']:.3f}, "
                  f"Studentized C={row['Studentized_C']:.3f}")
    
    # 6. 多证据层分析
    print("\n6. 多证据层批量分析...")
    
    # 创建证据层
    evidence_layers = {
        'Au_Anomaly': (processed_data['Au'] >= threshold_result['threshold']).astype(int),
        'As_Anomaly': (processed_data['As'] >= np.percentile(processed_data['As'], 90)).astype(int),
        'Sb_Anomaly': (processed_data['Sb'] >= np.percentile(processed_data['Sb'], 90)).astype(int)
    }
    
    # 批量计算证据权
    batch_results = woe_calculator.calculate_weights(
        evidence_layers,
        training_points
    )
    
    print(f"   批量计算完成，分析了 {len(batch_results)} 个证据层")
    
    # 显著性检验
    validation = woe_calculator.validate_significance(batch_results)
    print(f"   总体显著性: {validation['overall']['overall_significance_rate']:.2%}")
    
    # 7. 结果汇总与可视化
    print("\n7. 生成分析报告...")
    
    # 创建汇总统计
    summary_stats = woe_calculator.get_summary_statistics(batch_results)
    print("\n   证据层汇总:")
    for _, row in summary_stats.iterrows():
        print(f"     {row['Layer_Name']}: {row['Significant_Classes']}/{row['Total_Classes']} 显著类别, "
              f"最大对比度: {row['Max_Contrast']:.3f}")
    
    # 生成响应面
    response_surface = woe_calculator.create_response_surface(
        evidence_layers,
        batch_results
    )
    
    print(f"\n   响应面范围: {response_surface.min():.3f} - {response_surface.max():.3f}")
    
    # 8. 保存结果
    print("\n8. 保存分析结果...")
    
    # 创建输出目录
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    geochem_data.to_csv(f'{output_dir}/geochemical_data.csv', index=False)
    processed_data.to_csv(f'{output_dir}/processed_data.csv', index=False)
    clr_data.to_csv(f'{output_dir}/clr_data.csv', index=False)
    
    # 保存分析结果
    woe_result.to_csv(f'{output_dir}/au_weights_of_evidence.csv', index=False)
    summary_stats.to_csv(f'{output_dir}/evidence_layers_summary.csv', index=False)
    
    # 保存图形
    if 'figure' in r_mode_result:
        r_mode_result['figure'].savefig(f'{output_dir}/r_mode_clustering.png', dpi=300, bbox_inches='tight')
    
    if 'loadings_plot' in pca_result:
        pca_result['loadings_plot'].savefig(f'{output_dir}/pca_loadings.png', dpi=300, bbox_inches='tight')
    
    ca_plot.savefig(f'{output_dir}/ca_fractal_analysis.png', dpi=300, bbox_inches='tight')
    
    # 证据权结果图
    woe_plot = woe_calculator.plot_woe_results(woe_result, plot_type='contrast')
    woe_plot.savefig(f'{output_dir}/weights_of_evidence.png', dpi=300, bbox_inches='tight')
    
    print(f"   结果已保存到 {output_dir} 目录")
    
    # 9. 总结报告
    print("\n" + "=" * 60)
    print("分析总结报告")
    print("=" * 60)
    
    print(f"\n数据概况:")
    print(f"  - 样品数量: {len(geochem_data)}")
    print(f"  - 分析元素: {', '.join(elements)}")
    print(f"  - 矿点数量: {geochem_data['Is_Deposit'].sum()} ({geochem_data['Is_Deposit'].sum()/len(geochem_data):.1%})")
    
    print(f"\n元素组合分析:")
    print(f"  - 主要共生组合: Au-As-Sb-Hg (典型的卡林型金矿特征)")
    print(f"  - 推荐找矿元素: {', '.join(pca_result['recommended_elements'])}")
    
    print(f"\n异常分析:")
    print(f"  - Au异常阈值: {threshold_result['threshold']:.4f}")
    print(f"  - 异常样品: {threshold_result['n_anomalies']} ({threshold_result['anomaly_percentage']:.1f}%)")
    print(f"  - 异常样品中矿点比例: {geochem_data.loc[geochem_data['Au'] >= threshold_result['threshold'], 'Is_Deposit'].mean():.1%}")
    
    print(f"\n证据权分析:")
    significant_evidence = summary_stats[summary_stats['Significant_Classes'] > 0]
    if len(significant_evidence) > 0:
        print(f"  - 显著证据层: {', '.join(significant_evidence['Layer_Name'])}")
        best_layer = significant_evidence.loc[significant_evidence['Max_Contrast'].idxmax()]
        print(f"  - 最佳证据层: {best_layer['Layer_Name']} (对比度: {best_layer['Max_Contrast']:.3f})")
    else:
        print("  - 警告: 未发现显著的证据层")
    
    print(f"\n勘探建议:")
    if threshold_result['anomaly_percentage'] > 15:
        print("  - 异常范围较大，建议进一步缩小靶区")
    elif threshold_result['anomaly_percentage'] < 5:
        print("  - 异常范围较小，可能存在遗漏风险")
    else:
        print("  - 异常范围合理，可作为优先勘探靶区")
    
    if len(significant_evidence) > 0:
        print("  - 建议基于显著证据层开展详细地球化学测量")
        print("  - 优先验证高对比度区域的地质条件")
    else:
        print("  - 建议重新考虑元素组合或调整异常阈值")
    
    print("\n" + "=" * 60)
    print("工作流执行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()