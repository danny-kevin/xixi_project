"""
主入口文件
Main Entry Point

提供统一的命令行接口来运行训练、评估等任务
"""

import argparse
import sys
from pathlib import Path
import torch

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
        logger.info('Mode: eval')
        if not args.checkpoint:
            logger.error('Eval mode requires --checkpoint')
            sys.exit(1)

        try:
            from src.models.hybrid_model import ModelFactory
            from src.evaluation.metrics import RegressionMetrics
            from src.data.pipeline import DataPipeline

            pipeline = DataPipeline(
                data_dir=config.data.data_dir,
                window_size=config.data.window_size,
                horizon=config.data.prediction_horizon,
                train_ratio=config.data.train_ratio,
                val_ratio=config.data.val_ratio,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                use_single_file=True,
                single_file_name='dataset_US_final.csv',
                date_column='Date',
                target_column=config.data.target_column,
                group_column='State',
                one_hot_columns=['State'],
                normalize=False,
            )

            _, _, test_loader = pipeline.run()
            feature_names = pipeline.get_feature_names()
            config.data.feature_columns = feature_names
            config.model.num_variables = len(feature_names)
            config.model.prediction_horizon = config.data.prediction_horizon

            model = ModelFactory.create_model(config).to(device)

            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'[OK] loaded checkpoint: {args.checkpoint}')

            model.eval()
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    predictions = model(inputs)

                    if predictions.dim() == 3 and predictions.size(-1) == 1:
                        predictions = predictions.squeeze(-1)
                    if targets.dim() == 3 and targets.size(-1) == 1:
                        targets = targets.squeeze(-1)

                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())

            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)

            metrics = RegressionMetrics.compute_all(
                predictions_tensor.numpy(),
                targets_tensor.numpy(),
            )

            logger.info('Eval results:')
            logger.info(f'  MSE:  {metrics.mse:.4f}')
            logger.info(f'  RMSE: {metrics.rmse:.4f}')
            logger.info(f'  MAE:  {metrics.mae:.4f}')
            logger.info(f'  MAPE: {metrics.mape:.2f}%')
            logger.info(f'  R2:   {metrics.r2:.4f}')

        except Exception as e:
            logger.error(f'Eval failed: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.mode == 'predict':
        logger.info('Mode: predict')
        if not args.checkpoint:
            logger.error('Predict mode requires --checkpoint')
            sys.exit(1)

        try:
            from src.models.hybrid_model import ModelFactory
            from src.data.pipeline import DataPipeline

            pipeline = DataPipeline(
                data_dir=config.data.data_dir,
                window_size=config.data.window_size,
                horizon=config.data.prediction_horizon,
                train_ratio=config.data.train_ratio,
                val_ratio=config.data.val_ratio,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                use_single_file=True,
                single_file_name='dataset_US_final.csv',
                date_column='Date',
                target_column=config.data.target_column,
                group_column='State',
                one_hot_columns=['State'],
                normalize=False,
            )

            df = pipeline.load_data()
            _ = pipeline.preprocess_data(df)
            feature_names = pipeline.get_feature_names()
            config.data.feature_columns = feature_names
            config.model.num_variables = len(feature_names)
            config.model.prediction_horizon = config.data.prediction_horizon

            model = ModelFactory.create_model(config).to(device)

            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'[OK] loaded checkpoint: {args.checkpoint}')

            model.eval()
            logger.info('[OK] model is ready for prediction')
            logger.info('Use AttentionMTCNLSTM.predict() for forecasts')

        except Exception as e:
            logger.error(f'Predict init failed: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.mode == 'experiment':

        logger.info('模式: 完整实验')
        from run_experiment import run_experiment
        run_experiment(config, device, args.use_wandb)
    
    logger.info('='*80)
    logger.info('[OK] 任务完成')
    logger.info('='*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WARN]  用户中断')
        sys.exit(0)
    except Exception as e:
        print(f'\n[ERR] 错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
