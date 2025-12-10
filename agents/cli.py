"""
Gold-Seeker 命令行接口

提供命令行工具，支持快速执行地球化学分析任务。
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd

from . import __version__, print_platform_info
from .config import load_config
from .utils import setup_logging, validate_geochemical_data
from .spatial_analyst import SpatialAnalystAgent
from .tools.geochem import GeochemSelector, GeochemProcessor, FractalAnomalyFilter, WeightsOfEvidenceCalculator


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="gold-seeker",
        description="Gold-Seeker: 地球化学找矿预测智能平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  gold-seeker analyze data.csv --elements Au As Sb Hg
  gold-seeker workflow data.csv --config config.yaml
  gold-seeker validate data.csv --elements Au As Sb
  gold-seeker info
        """
    )
    
    # 版本信息
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Gold-Seeker v{__version__}"
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 分析命令
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="执行地球化学数据分析"
    )
    analyze_parser.add_argument(
        "data_file",
        help="地球化学数据文件路径"
    )
    analyze_parser.add_argument(
        "--elements", "-e",
        nargs="+",
        default=["Au", "As", "Sb", "Hg"],
        help="要分析的元素列表"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        default="output",
        help="输出目录"
    )
    analyze_parser.add_argument(
        "--config", "-c",
        help="配置文件路径"
    )
    analyze_parser.add_argument(
        "--training-points",
        help="训练点文件路径"
    )
    
    # 工作流命令
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="执行完整工作流"
    )
    workflow_parser.add_argument(
        "data_file",
        help="地球化学数据文件路径"
    )
    workflow_parser.add_argument(
        "--config", "-c",
        help="配置文件路径"
    )
    workflow_parser.add_argument(
        "--output", "-o",
        default="output",
        help="输出目录"
    )
    
    # 验证命令
    validate_parser = subparsers.add_parser(
        "validate",
        help="验证数据质量"
    )
    validate_parser.add_argument(
        "data_file",
        help="地球化学数据文件路径"
    )
    validate_parser.add_argument(
        "--elements", "-e",
        nargs="+",
        required=True,
        help="要验证的元素列表"
    )
    validate_parser.add_argument(
        "--config", "-c",
        help="配置文件路径"
    )
    
    # 信息命令
    info_parser = subparsers.add_parser(
        "info",
        help="显示平台信息"
    )
    
    # 示例命令
    example_parser = subparsers.add_parser(
        "example",
        help="运行示例分析"
    )
    example_parser.add_argument(
        "--type", "-t",
        choices=["synthetic", "workflow"],
        default="synthetic",
        help="示例类型"
    )
    
    return parser


def cmd_analyze(args) -> int:
    """执行分析命令"""
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        logger = setup_logging(
            level=config.get_log_level(),
            log_file=Path(args.output) / "analysis.log"
        )
        
        logger.info(f"开始分析数据: {args.data_file}")
        
        # 加载数据
        data = pd.read_csv(args.data_file)
        logger.info(f"数据加载完成: {data.shape}")
        
        # 验证数据
        detection_limits = config.get_detection_limits()
        valid, errors = validate_geochemical_data(
            data, args.elements, detection_limits
        )
        
        if not valid:
            logger.error("数据验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
        
        # 创建分析器
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(**config.get_llm_config())
        analyst = SpatialAnalystAgent(llm, detection_limits)
        
        # 加载训练点（如果提供）
        training_points = None
        if args.training_points:
            training_points = pd.read_csv(args.training_points)
            logger.info(f"训练点加载完成: {training_points.shape}")
        
        # 执行分析
        result = analyst.analyze_geochemical_data(
            data=data,
            elements=args.elements,
            training_points=training_points
        )
        
        # 生成报告
        report = analyst.generate_analysis_report(result)
        
        # 保存结果
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存报告
        report_file = output_dir / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存结果数据
        if hasattr(result, 'to_dict'):
            import json
            result_file = output_dir / "analysis_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析完成，结果保存到: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"分析失败: {e}")
        return 1


def cmd_workflow(args) -> int:
    """执行工作流命令"""
    try:
        # 运行完整工作流示例
        from examples.complete_workflow import main as workflow_main
        
        # 设置参数
        import sys
        sys.argv = [
            "complete_workflow.py",
            "--output", args.output
        ]
        
        if args.config:
            sys.argv.extend(["--config", args.config])
        
        return workflow_main()
        
    except Exception as e:
        print(f"工作流执行失败: {e}")
        return 1


def cmd_validate(args) -> int:
    """执行验证命令"""
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        logger = setup_logging(level=config.get_log_level())
        
        logger.info(f"验证数据: {args.data_file}")
        
        # 加载数据
        data = pd.read_csv(args.data_file)
        logger.info(f"数据加载完成: {data.shape}")
        
        # 验证数据
        detection_limits = config.get_detection_limits()
        valid, errors = validate_geochemical_data(
            data, args.elements, detection_limits
        )
        
        if valid:
            print("✅ 数据验证通过")
            return 0
        else:
            print("❌ 数据验证失败:")
            for error in errors:
                print(f"  - {error}")
            return 1
        
    except Exception as e:
        print(f"验证失败: {e}")
        return 1


def cmd_info(args) -> int:
    """执行信息命令"""
    print_platform_info()
    return 0


def cmd_example(args) -> int:
    """执行示例命令"""
    try:
        if args.type == "synthetic":
            # 运行合成数据示例
            from examples.complete_workflow import generate_synthetic_data
            from examples.complete_workflow import main as workflow_main
            
            print("🔬 运行合成数据示例...")
            return workflow_main()
        
        elif args.type == "workflow":
            # 运行工作流示例
            from examples.complete_workflow import main as workflow_main
            
            print("🔄 运行完整工作流示例...")
            return workflow_main()
        
        return 0
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """主函数"""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # 如果没有提供命令，显示帮助
    if not args.command:
        parser.print_help()
        return 1
    
    # 执行对应命令
    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "workflow":
        return cmd_workflow(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "example":
        return cmd_example(args)
    else:
        print(f"未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())