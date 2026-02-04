"""
主入口文件
Main Entry Point

提供统一的命令行接口来运行训练、评估等任务
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config, Config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.device_manager import DeviceManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='注意力增强M-TCN-LSTM传染病预测模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础参数
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'predict', 'experiment'],
        default='train',
        help='运行模式'
    )
    
    # 训练参数覆盖
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--device', type=str, help='设备 (cuda/cpu)')
    
    # 路径参数
    parser.add_argument('--checkpoint', type=str, help='检查点路径（用于恢复训练或评估）')
    parser.add_argument('--output-dir', type=str, default='results', help='输出目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--use-wandb', action='store_true', help='使用Weights & Biases追踪')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logger(
        'main',
        log_file=f'logs/main.log',
        level=args.log_level
    )
    
    logger.info('='*80)
    logger.info('注意力增强M-TCN-LSTM传染病预测模型')
    logger.info('='*80)
    
    # 加载配置
    logger.info(f'加载配置文件: {args.config}')
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.training.device = args.device
    if args.seed:
        config.seed = args.seed
    
    # 设置随机种子
    logger.info(f'设置随机种子: {config.seed}')
    set_seed(config.seed)
    
    # 设置设备
    device_manager = DeviceManager()
    device = device_manager.get_device(config.training.device)
    logger.info(f'使用设备: {device}')
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'输出目录: {output_dir}')
    
    # 根据模式执行不同任务
    if args.mode == 'train':
        logger.info('模式: 训练')
        from train import train
        train(config, device, args.checkpoint, args.use_wandb)
        
    elif args.mode == 'eval':
        logger.info('模式: 评估')
        if not args.checkpoint:
            logger.error('评估模式需要指定 --checkpoint 参数')
            sys.exit(1)
        
        try:
            from src.models.hybrid_model import ModelFactory
            from src.evaluation.metrics import RegressionMetrics
            from load_us_data import USCovidDataLoader
            
            # 加载数据
            data_path = f"{config.data.raw_dir}/dataset_US_final.csv"
            loader = USCovidDataLoader(data_path=data_path)
            data_dict = loader.prepare_data(
                target_column=config.data.target_column,
                window_size=config.data.window_size,
                horizon=config.data.prediction_horizon,
                train_ratio=config.data.train_ratio,
                val_ratio=config.data.val_ratio,
                scaler_type=config.data.scaler_type
            )
            _, _, test_loader = loader.create_dataloaders(
                data_dict,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers
            )
            
            # 创建模型
            model = ModelFactory.create_model(config).to(device)
            
            # 加载检查点
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'✅ 模型已从 {args.checkpoint} 加载')
            
            # 评估
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    predictions = model(inputs)
                    
                    # 处理维度
                    if predictions.dim() == 3 and predictions.size(-1) == 1:
                        predictions = predictions.squeeze(-1)
                    if targets.dim() == 3 and targets.size(-1) == 1:
                        targets = targets.squeeze(-1)
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
            
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # 计算指标
            metrics = RegressionMetrics.compute_all_metrics(
                predictions_tensor.numpy(),
                targets_tensor.numpy()
            )
            
            logger.info('评估结果:')
            logger.info(f'  MSE:  {metrics["mse"]:.4f}')
            logger.info(f'  RMSE: {metrics["rmse"]:.4f}')
            logger.info(f'  MAE:  {metrics["mae"]:.4f}')
            logger.info(f'  MAPE: {metrics["mape"]:.2f}%')
            logger.info(f'  R²:   {metrics["r2"]:.4f}')
            
        except Exception as e:
            logger.error(f'评估过程出错: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    elif args.mode == 'predict':
        logger.info('模式: 预测')
        if not args.checkpoint:
            logger.error('预测模式需要指定 --checkpoint 参数')
            sys.exit(1)
        
        try:
            from src.models.hybrid_model import ModelFactory
            
            # 创建模型
            model = ModelFactory.create_model(config).to(device)
            
            # 加载检查点
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'✅ 模型已从 {args.checkpoint} 加载')
            
            model.eval()
            logger.info('✅ 模型已就绪，可用于预测')
            logger.info('注意: 详细的预测功能请使用 src.models.hybrid_model.AttentionMTCNLSTM.predict() 方法')
            
        except Exception as e:
            logger.error(f'预测模式初始化失败: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    elif args.mode == 'experiment':
        logger.info('模式: 完整实验')
        from run_experiment import run_experiment
        run_experiment(config, device, args.use_wandb)
    
    logger.info('='*80)
    logger.info('✅ 任务完成')
    logger.info('='*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n⚠️  用户中断')
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ 错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
