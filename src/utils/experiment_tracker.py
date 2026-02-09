"""
实验追踪器
Experiment Tracker

集成TensorBoard和Weights & Biases进行实验追踪
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

# 尝试导入TensorBoard
try:
  from torch.utils.tensorboard import SummaryWriter
  TENSORBOARD_AVAILABLE = True
except ImportError:
  TENSORBOARD_AVAILABLE = False
  logger.warning('TensorBoard不可用，请安装: pip install tensorboard')

# 尝试导入Weights & Biases
try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  logger.warning('Weights & Biases不可用，请安装: pip install wandb')


class ExperimentTracker:
  """
  实验追踪器
  
  统一管理TensorBoard和Weights & Biases的日志记录
  """
  
  def __init__(
    self,
    experiment_name: str,
    config: Optional[Any] = None,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    log_dir: str = 'logs',
    wandb_project: Optional[str] = None
  ):
    """
    初始化实验追踪器
    
    Args:
      experiment_name: 实验名称
      config: 配置对象
      use_tensorboard: 是否使用TensorBoard
      use_wandb: 是否使用Weights & Biases
      log_dir: 日志目录（TensorBoard）
      wandb_project: W&B项目名称
    """
    self.experiment_name = experiment_name
    self.config = config
    
    # TensorBoard
    self.tensorboard_writer = None
    if use_tensorboard and TENSORBOARD_AVAILABLE:
      log_path = Path(log_dir) / experiment_name
      log_path.mkdir(parents=True, exist_ok=True)
      self.tensorboard_writer = SummaryWriter(log_dir=str(log_path))
      logger.info(f' TensorBoard已启用: {log_path}')
    elif use_tensorboard and not TENSORBOARD_AVAILABLE:
      logger.warning(' TensorBoard未安装，跳过')
    
    # Weights & Biases
    self.wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
      # 准备配置字典
      config_dict = {}
      if config is not None:
        if hasattr(config, 'to_dict'):
          config_dict = config.to_dict()
        elif isinstance(config, dict):
          config_dict = config
      
      # 初始化W&B
      self.wandb_run = wandb.init(
        project=wandb_project or 'epidemic_prediction',
        name=experiment_name,
        config=config_dict
      )
      logger.info(f' Weights & Biases已启用')
    elif use_wandb and not WANDB_AVAILABLE:
      logger.warning(' Weights & Biases未安装，跳过')
    
    self.step = 0
  
  def log_metric(
    self,
    name: str,
    value: Union[float, int, None],
    step: Optional[int] = None
  ) -> None:
    """
    记录单个指标
    
    Args:
      name: 指标名称
      value: 指标值
      step: 步数（None表示使用内部计数器）
    """
    if step is None:
      step = self.step

    # Optional metrics may be None (e.g. CRPS not computed). Skip in that case.
    if value is None:
      return
    
    # TensorBoard
    if self.tensorboard_writer is not None:
      self.tensorboard_writer.add_scalar(name, value, step)
    
    # Weights & Biases
    if self.wandb_run is not None:
      wandb.log({name: value}, step=step)
  
  def log_metrics(
    self,
    metrics: Dict[str, Union[float, int, None]],
    step: Optional[int] = None
  ) -> None:
    """
    记录多个指标
    
    Args:
      metrics: 指标字典
      step: 步数
    """
    if step is None:
      step = self.step
    
    # TensorBoard
    if self.tensorboard_writer is not None:
      for name, value in metrics.items():
        if value is None:
          continue
        if isinstance(value, (list, tuple)):
          # 如果是列表，记录最后一个值
          value = value[-1] if len(value) > 0 else 0
        self.tensorboard_writer.add_scalar(name, value, step)
    
    # Weights & Biases
    if self.wandb_run is not None:
      # 处理列表值
      processed_metrics = {}
      for name, value in metrics.items():
        if value is None:
          continue
        if isinstance(value, (list, tuple)):
          value = value[-1] if len(value) > 0 else 0
        processed_metrics[name] = value
      wandb.log(processed_metrics, step=step)
    
    self.step += 1
  
  def log_histogram(
    self,
    name: str,
    values: torch.Tensor,
    step: Optional[int] = None
  ) -> None:
    """
    记录直方图
    
    Args:
      name: 名称
      values: 数值张量
      step: 步数
    """
    if step is None:
      step = self.step
    
    # TensorBoard
    if self.tensorboard_writer is not None:
      self.tensorboard_writer.add_histogram(name, values, step)
    
    # Weights & Biases
    if self.wandb_run is not None:
      wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
  
  def log_image(
    self,
    name: str,
    image: torch.Tensor,
    step: Optional[int] = None
  ) -> None:
    """
    记录图像
    
    Args:
      name: 名称
      image: 图像张量
      step: 步数
    """
    if step is None:
      step = self.step
    
    # TensorBoard
    if self.tensorboard_writer is not None:
      self.tensorboard_writer.add_image(name, image, step)
    
    # Weights & Biases
    if self.wandb_run is not None:
      wandb.log({name: wandb.Image(image.cpu().numpy())}, step=step)
  
  def log_text(
    self,
    name: str,
    text: str,
    step: Optional[int] = None
  ) -> None:
    """
    记录文本
    
    Args:
      name: 名称
      text: 文本内容
      step: 步数
    """
    if step is None:
      step = self.step
    
    # TensorBoard
    if self.tensorboard_writer is not None:
      self.tensorboard_writer.add_text(name, text, step)
    
    # Weights & Biases
    if self.wandb_run is not None:
      wandb.log({name: text}, step=step)
  
  def watch_model(self, model: torch.nn.Module) -> None:
    """
    监控模型（仅W&B）
    
    Args:
      model: 模型
    """
    if self.wandb_run is not None:
      wandb.watch(model, log='all', log_freq=100)
      logger.info(' 模型监控已启用')
  
  def finish(self) -> None:
    """结束追踪"""
    # 关闭TensorBoard
    if self.tensorboard_writer is not None:
      self.tensorboard_writer.close()
      logger.info('TensorBoard已关闭')
    
    # 关闭Weights & Biases
    if self.wandb_run is not None:
      wandb.finish()
      logger.info('Weights & Biases已关闭')


# 使用示例
if __name__ == '__main__':
  # 设置日志
  logging.basicConfig(level=logging.INFO)
  
  # 创建追踪器
  tracker = ExperimentTracker(
    experiment_name='test_experiment',
    use_tensorboard=True,
    use_wandb=False # 设置为True需要先登录wandb
  )
  
  # 记录指标
  for epoch in range(10):
    tracker.log_metrics({
      'train_loss': 1.0 / (epoch + 1),
      'val_loss': 1.2 / (epoch + 1),
      'learning_rate': 0.001 * (0.9 ** epoch)
    }, step=epoch)
  
  # 记录直方图
  weights = torch.randn(100)
  tracker.log_histogram('model/weights', weights, step=0)
  
  # 结束
  tracker.finish()
  
  print('\n 实验追踪器测试完成')
  print('运行 tensorboard --logdir logs 查看结果')
