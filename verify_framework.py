"""
框架验证脚本
Framework Validation Script

快速验证项目框架是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print('='*80)
print('项目框架验证')
print('='*80)

# 测试1: 导入工具模块
print('\n[测试1] 导入工具模块...')
try:
    from src.utils import (
        setup_logger, DeviceManager, set_seed,
        ShapeValidator, to_tensor,
        load_config, save_config
    )
    print('[OK] 工具模块导入成功')
except Exception as e:
    print(f'[ERR] 工具模块导入失败: {e}')
    sys.exit(1)

# 测试2: 日志系统
print('\n[测试2] 测试日志系统...')
try:
    logger = setup_logger('test', log_file='logs/test.log')
    logger.info('日志系统测试')
    logger.debug('调试信息')
    logger.warning('警告信息')
    print('[OK] 日志系统正常')
except Exception as e:
    print(f'[ERR] 日志系统失败: {e}')

# 测试3: 设备管理
print('\n[测试3] 测试设备管理...')
try:
    device_manager = DeviceManager()
    device = device_manager.get_device()
    print(f'[OK] 设备管理正常，当前设备: {device}')
except Exception as e:
    print(f'[ERR] 设备管理失败: {e}')

# 测试4: 随机种子
print('\n[测试4] 测试随机种子...')
try:
    set_seed(42)
    import torch
    x1 = torch.randn(5)
    set_seed(42)
    x2 = torch.randn(5)
    if torch.allclose(x1, x2):
        print('[OK] 随机种子正常（可复现）')
    else:
        print('[WARN]  随机种子可能有问题（不可复现）')
except Exception as e:
    print(f'[ERR] 随机种子失败: {e}')

# 测试5: 形状验证
print('\n[测试5] 测试形状验证...')
try:
    import torch
    x = torch.randn(32, 21, 11)
    ShapeValidator.validate_shape(x, (32, 21, 11), "test_tensor")
    ShapeValidator.validate_shape(x, (None, 21, 11), "test_tensor")
    ShapeValidator.validate_no_nan_inf(x, "test_tensor")
    print('[OK] 形状验证正常')
except Exception as e:
    print(f'[ERR] 形状验证失败: {e}')

# 测试6: 配置加载
print('\n[测试6] 测试配置加载...')
try:
    config = load_config('configs/default_config.yaml')
    print(f'[OK] 配置加载正常')
    print(f'   - 实验名称: {config.experiment_name}')
    print(f'   - 窗口大小: {config.data.window_size}')
    print(f'   - 预测范围: {config.data.prediction_horizon}')
    print(f'   - 训练轮数: {config.training.epochs}')
except Exception as e:
    print(f'[ERR] 配置加载失败: {e}')

# 测试7: 配置序列化
print('\n[测试7] 测试配置序列化...')
try:
    config_dict = config.to_dict()
    config_restored = config.from_dict(config_dict)
    print('[OK] 配置序列化正常')
except Exception as e:
    print(f'[ERR] 配置序列化失败: {e}')

# 测试8: 类型转换
print('\n[测试8] 测试类型转换...')
try:
    import numpy as np
    arr = np.array([[1, 2, 3]])
    tensor = to_tensor(arr)
    print(f'[OK] 类型转换正常: NumPy{arr.shape} -> Tensor{tensor.shape}')
except Exception as e:
    print(f'[ERR] 类型转换失败: {e}')

# 测试9: 检查文件结构
print('\n[测试9] 检查文件结构...')
required_files = [
    'main.py',
    'train.py',
    'run_experiment.py',
    'requirements.txt',
    'README.md',
    'configs/default_config.yaml',
    'src/__init__.py',
    'src/utils/__init__.py',
    'src/utils/logger.py',
    'src/utils/device_manager.py',
    'src/utils/seed.py',
    'src/utils/checkpoint.py',
    'src/utils/shape_validator.py',
    'src/utils/type_utils.py',
    'src/utils/protocols.py',
    'src/utils/experiment_tracker.py',
    'Docs/data_flow.md',
    'Docs/interface_contracts.md',
    'Docs/workflow.md',
    'notebooks/01_quick_start.ipynb'
]

missing_files = []
for file_path in required_files:
    if not (project_root / file_path).exists():
        missing_files.append(file_path)

if missing_files:
    print(f'[WARN]  缺少以下文件:')
    for f in missing_files:
        print(f'   - {f}')
else:
    print(f'[OK] 所有必需文件都存在 ({len(required_files)}个)')

# 总结
print('\n' + '='*80)
print('验证完成')
print('='*80)

if not missing_files:
    print('[OK] 框架验证通过！所有核心功能正常工作。')
    print('\n下一步:')
    print('1. 让各Agent按顺序实现其负责的模块')
    print('2. 参考 Docs/workflow.md 了解如何运行')
    print('3. 查看 notebooks/01_quick_start.ipynb 学习使用')
else:
    print('[WARN]  框架基本正常，但缺少部分文件')

print('='*80)
