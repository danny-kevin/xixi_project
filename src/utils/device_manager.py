"""
设备管理器
Device Manager

自动检测和管理计算设备（CPU/CUDA/MPS）
"""

import torch
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    设备管理器
    
    自动检测可用的计算设备并提供统一的接口
    """
    
    def __init__(self):
        """初始化设备管理器"""
        self._device = None
        self._device_name = None
        self._detect_device()
    
    def _detect_device(self) -> None:
        """检测可用的设备"""
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            self._device_name = torch.cuda.get_device_name(0)
            logger.info(f'检测到CUDA设备: {self._device_name}')
            logger.info(f'CUDA版本: {torch.version.cuda}')
            logger.info(f'可用GPU数量: {torch.cuda.device_count()}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon (M1/M2) GPU
            self._device = torch.device('mps')
            self._device_name = 'Apple Silicon GPU'
            logger.info(f'检测到MPS设备: {self._device_name}')
        else:
            self._device = torch.device('cpu')
            self._device_name = 'CPU'
            logger.info('使用CPU设备')
    
    def get_device(self, preferred: Optional[str] = None) -> torch.device:
        """
        获取计算设备
        
        Args:
            preferred: 首选设备 ('cuda', 'mps', 'cpu')
                      如果首选设备不可用，会自动降级
        
        Returns:
            torch.device对象
        """
        if preferred is None:
            return self._device
        
        preferred = preferred.lower()
        
        if preferred == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                logger.warning('CUDA不可用，降级到CPU')
                return torch.device('cpu')
        
        elif preferred == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                logger.warning('MPS不可用，降级到CPU')
                return torch.device('cpu')
        
        elif preferred == 'cpu':
            return torch.device('cpu')
        
        else:
            logger.warning(f'未知设备类型: {preferred}，使用默认设备')
            return self._device
    
    def get_device_name(self) -> str:
        """获取设备名称"""
        return self._device_name
    
    def get_memory_info(self) -> dict:
        """
        获取设备内存信息
        
        Returns:
            包含内存信息的字典
        """
        if self._device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        else:
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0
            }
    
    def clear_cache(self) -> None:
        """清空GPU缓存"""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info('GPU缓存已清空')
    
    def set_device(self, device_id: int = 0) -> None:
        """
        设置当前使用的GPU
        
        Args:
            device_id: GPU设备ID
        """
        if self._device.type == 'cuda':
            torch.cuda.set_device(device_id)
            logger.info(f'切换到GPU {device_id}')
    
    def print_info(self) -> None:
        """打印设备信息"""
        print('='*60)
        print('设备信息')
        print('='*60)
        print(f'设备类型: {self._device.type}')
        print(f'设备名称: {self._device_name}')
        
        if self._device.type == 'cuda':
            print(f'CUDA版本: {torch.version.cuda}')
            print(f'cuDNN版本: {torch.backends.cudnn.version()}')
            print(f'GPU数量: {torch.cuda.device_count()}')
            
            mem_info = self.get_memory_info()
            print(f'已分配内存: {mem_info["allocated"]:.2f} GB')
            print(f'已保留内存: {mem_info["reserved"]:.2f} GB')
            print(f'最大分配内存: {mem_info["max_allocated"]:.2f} GB')
        
        print('='*60)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    快捷函数：获取计算设备
    
    Args:
        preferred: 首选设备
        
    Returns:
        torch.device对象
    """
    manager = DeviceManager()
    return manager.get_device(preferred)


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建设备管理器
    device_manager = DeviceManager()
    
    # 打印设备信息
    device_manager.print_info()
    
    # 获取设备
    device = device_manager.get_device()
    print(f'\n当前设备: {device}')
    
    # 测试张量
    x = torch.randn(10, 10).to(device)
    print(f'张量设备: {x.device}')
    
    # 获取内存信息
    if device.type == 'cuda':
        mem_info = device_manager.get_memory_info()
        print(f'\n内存使用: {mem_info["allocated"]:.4f} GB')
    
    print('\n[OK] 设备管理器测试完成')
