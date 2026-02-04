"""
完整实验流程脚本
Complete Experiment Pipeline

演示从数据准备到模型评估的完整流程
"""

import torch
from pathlib import Path
from typing import Optional

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.visualization import Visualizer


def run_experiment(
    config: Config,
    device: torch.device,
    use_wandb: bool = False
):
    """
    运行完整实验流程
    
    流程:
    1. 数据准备
    2. 模型创建
    3. 模型训练
    4. 模型评估
    5. 可解释性分析
    6. 结果保存
    
    Args:
        config: 配置对象
        device: 计算设备
        use_wandb: 是否使用Weights & Biases
    """
    logger = setup_logger('experiment', log_file='logs/experiment.log')
    
    logger.info('='*80)
    logger.info('开始完整实验流程')
    logger.info('='*80)
    
    # 创建输出目录
    output_dir = Path('results') / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'输出目录: {output_dir}')
    
    # 创建实验追踪器
    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        config=config,
        use_wandb=use_wandb,
        log_dir='logs/tensorboard'
    )
    
    # 创建可视化器
    visualizer = Visualizer(save_dir=output_dir / 'figures')
    
    try:
        # ==================== 阶段1: 数据准备 ====================
        logger.info('\n' + '='*80)
        logger.info('阶段1: 数据准备')
        logger.info('='*80)
        
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

        train_loader, val_loader, test_loader = pipeline.run()
        feature_names = pipeline.get_feature_names()
        config.data.feature_columns = feature_names
        config.model.num_variables = len(feature_names)
        config.model.prediction_horizon = config.data.prediction_horizon

        logger.info('[OK] data ready')
        logger.info(f'  Train samples: {len(train_loader.dataset)}')
        logger.info(f'  Val samples: {len(val_loader.dataset)}')
        logger.info(f'  Test samples: {len(test_loader.dataset)}')
        # ==================== 阶段2: 模型创建 ====================
        logger.info('\n' + '='*80)
        logger.info('阶段2: 模型创建')
        logger.info('='*80)
        
        from src.models.hybrid_model import ModelFactory
        
        # 使用ModelFactory创建模型
        model = ModelFactory.create_model(config).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f'[OK] 模型创建完成，参数量: {num_params:,}')
        logger.info(f'  输入变量数: {config.model.num_variables}')
        logger.info(f'  TCN通道: {config.model.tcn_channels}')
        logger.info(f'  LSTM隐藏层: {config.model.lstm_hidden_size}')
        logger.info(f'  预测范围: {config.model.prediction_horizon}天')
        
        # ==================== 阶段3: 模型训练 ====================
        logger.info('\n' + '='*80)
        logger.info('阶段3: 模型训练')
        logger.info('='*80)
        
        from src.training import Trainer, TrainingConfig
        from src.training.loss import HybridLoss
        from src.training.scheduler import WarmupCosineScheduler
        
        # 创建训练配置
        training_config = TrainingConfig(
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            device=str(device),
            checkpoint_dir=config.training.checkpoint_dir,
            early_stopping_patience=config.training.early_stopping_patience,
            warmup_epochs=config.training.warmup_epochs,
            gradient_clip_val=config.training.gradient_clip_val,
            pretrain_epochs=config.training.pretrain_epochs if config.training.use_pretrain else 0,
            finetune_lr_ratio=config.training.finetune_lr_ratio
        )
        
        # 创建优化器、调度器和损失函数
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            min_lr=config.training.min_lr
        )
        
        criterion = HybridLoss()
        
        # 创建训练器
        trainer = Trainer(model, training_config, optimizer, scheduler, criterion)
        
        logger.info('开始训练...')
        history = trainer.train(train_loader, val_loader)
        
        logger.info('[OK] 训练完成')
        
        # 保存训练历史
        tracker.log_metrics(history)
        
        # 可视化训练曲线
        fig = visualizer.plot_training_history(
            history,
            save_name='training_history'
        )
        logger.info('[OK] 训练曲线已保存')
        
        # 保存模型
        checkpoint_path = output_dir / 'best_model.pth'
        trainer.save_checkpoint(str(checkpoint_path), is_best=True)
        logger.info(f'[OK] 模型已保存: {checkpoint_path}')
        
        # ==================== 阶段4: 模型评估 ====================
        logger.info('\n' + '='*80)
        logger.info('阶段4: 模型评估')
        logger.info('='*80)
        
        try:
            from src.evaluation.metrics import RegressionMetrics
            
            # 在测试集上评估
            logger.info('在测试集上评估...')
            test_loss = trainer.validate(test_loader)
            logger.info(f'测试集损失: {test_loss:.4f}')
            
            # 获取预测结果
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
            
            # 计算评估指标
            metrics = RegressionMetrics.compute_all(
                predictions_tensor.numpy(),
                targets_tensor.numpy()
            )
            
            logger.info('Eval results:')
            logger.info(f'  MSE:  {metrics.mse:.4f}')
            logger.info(f'  RMSE: {metrics.rmse:.4f}')
            logger.info(f'  MAE:  {metrics.mae:.4f}')
            logger.info(f'  MAPE: {metrics.mape:.2f}%')
            logger.info(f'  R2:   {metrics.r2:.4f}')
            
            # 记录到追踪器
            tracker.log_metrics(metrics.__dict__)
            
            # 可视化预测结果
            fig = visualizer.plot_predictions(
                actual=targets_tensor.numpy(),
                predicted=predictions_tensor.numpy(),
                save_name='predictions'
            )
            logger.info('[OK] 预测结果图已保存')
            
            # 残差分析
            fig = visualizer.plot_residuals(
                actual=targets_tensor.numpy(),
                predicted=predictions_tensor.numpy(),
                save_name='residuals'
            )
            logger.info('[OK] 残差分析图已保存')
            
        except ImportError as e:
            logger.warning(f'[WARN]  评估模块部分功能不可用: {e}')
            logger.info('基本评估：测试集损失已记录')
            tracker.log_metric('test_loss', test_loss)
        
        # ==================== 阶段5: 可解释性分析 ====================
        logger.info('\n' + '='*80)
        logger.info('阶段5: 可解释性分析')
        logger.info('='*80)
        
        try:
            from src.evaluation.interpretability import AttentionVisualizer
            
            # 注意力可视化
            att_visualizer = AttentionVisualizer(model, device=str(device))
            
            # 获取一个样本
            sample_input = next(iter(test_loader))[0][:1].to(device)
            attention_weights = att_visualizer.extract_attention_weights(sample_input)
            
            # 时间注意力热力图
            if 'temporal' in attention_weights or 'self_attention' in attention_weights:
                att_key = 'temporal' if 'temporal' in attention_weights else 'self_attention'
                fig = att_visualizer.plot_temporal_attention(
                    attention_weights[att_key],
                    save_path=str(output_dir / 'figures' / 'temporal_attention.png')
                )
                logger.info('[OK] 时间注意力图已保存')
            
            # 变量注意力
            if 'variable' in attention_weights or 'variable_attention' in attention_weights:
                att_key = 'variable' if 'variable' in attention_weights else 'variable_attention'
                try:
                    fig = att_visualizer.plot_variable_attention(
                        attention_weights[att_key],
                        variable_names=config.data.feature_columns,
                        save_path=str(output_dir / 'figures' / 'variable_attention.png')
                    )
                    logger.info('[OK] 变量注意力图已保存')
                except Exception as e:
                    logger.warning(f'变量注意力可视化失败: {e}')
            
            logger.info('[OK] 可解释性分析完成')
            
        except ImportError as e:
            logger.warning(f'[WARN]  可解释性分析模块不可用: {e}')
            logger.info('跳过可解释性分析')
        except Exception as e:
            logger.warning(f'[WARN]  可解释性分析出错: {e}')
            logger.info('跳过可解释性分析')
        
        # ==================== 阶段6: 保存配置和总结 ====================
        logger.info('\n' + '='*80)
        logger.info('阶段6: 保存配置和总结')
        logger.info('='*80)
        
        # 保存配置
        from src.utils.config import save_config
        config_path = output_dir / 'config.yaml'
        save_config(config, config_path)
        logger.info(f'[OK] 配置已保存: {config_path}')
        
        # 生成总结
        summary_path = output_dir / 'experiment_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('实验总结\n')
            f.write('='*80 + '\n\n')
            f.write(f'实验名称: {config.experiment_name}\n')
            f.write(f'随机种子: {config.seed}\n')
            f.write(f'设备: {device}\n\n')
            
            f.write('数据配置:\n')
            f.write(f'  窗口大小: {config.data.window_size}\n')
            f.write(f'  预测范围: {config.data.prediction_horizon}\n')
            f.write(f'  变量数量: {config.model.num_variables}\n\n')
            
            f.write('模型配置:\n')
            f.write(f'  TCN通道: {config.model.tcn_channels}\n')
            f.write(f'  注意力维度: {config.model.attention_embed_dim}\n')
            f.write(f'  LSTM隐藏层: {config.model.lstm_hidden_size}\n\n')
            
            f.write('训练配置:\n')
            f.write(f'  训练轮数: {config.training.epochs}\n')
            f.write(f'  批次大小: {config.training.batch_size}\n')
            f.write(f'  学习率: {config.training.learning_rate}\n\n')
            
            f.write('输出文件:\n')
            f.write(f'  模型: {output_dir}/best_model.pth\n')
            f.write(f'  配置: {output_dir}/config.yaml\n')
            f.write(f'  报告: {output_dir}/evaluation_report.txt\n')
            f.write(f'  图表: {output_dir}/figures/\n')
        
        logger.info(f'[OK] 实验总结已保存: {summary_path}')
        
    except Exception as e:
        logger.error(f'实验过程出错: {e}')
        raise
    
    finally:
        # 关闭追踪器
        tracker.finish()
        
        logger.info('\n' + '='*80)
        logger.info('[OK] 实验流程完成')
        logger.info(f'所有结果已保存到: {output_dir}')
        logger.info('='*80)


if __name__ == '__main__':
    """
    直接运行此脚本的示例
    
    使用方法:
        python run_experiment.py
    
    或使用main.py:
        python main.py --mode experiment --config configs/default_config.yaml
    """
    from src.utils.config import load_config
    from src.utils.device_manager import DeviceManager
    from src.utils.seed import set_seed
    
    # 加载配置
    config = load_config('configs/default_config.yaml')
    
    # 设置种子
    set_seed(config.seed)
    
    # 获取设备
    device_manager = DeviceManager()
    device = device_manager.get_device()
    
    # 运行实验
    run_experiment(config, device, use_wandb=False)
