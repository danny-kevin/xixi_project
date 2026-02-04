"""
测试数据预处理管道
Test Data Preprocessing Pipeline

验证数据加载、预处理和数据集创建功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_data, DataPipeline


def test_data_pipeline():
    """测试完整的数据预处理管道"""
    
    print("\n" + "=" * 70)
    print("🧪 测试数据预处理管道")
    print("=" * 70 + "\n")
    
    # 数据目录
    data_dir = project_root / 'data'
    
    # 检查数据是否存在
    raw_dir = data_dir / 'raw'
    if not raw_dir.exists() or not any(raw_dir.glob('*.csv')):
        print("⚠️  未找到数据文件，正在生成示例数据...")
        from scripts.generate_sample_data import generate_all_sample_data
        generate_all_sample_data(output_dir=str(raw_dir))
        print()
    
    try:
        # 方式1: 使用便捷函数
        print("📌 方式1: 使用 prepare_data() 便捷函数\n")
        train_loader, val_loader, test_loader, preprocessor = prepare_data(
            data_dir=str(data_dir),
            window_size=21,
            horizon=7,
            batch_size=32,
            num_workers=0  # Windows上设置为0
        )
        
        print("\n" + "=" * 70)
        print("✅ 数据准备成功！")
        print("=" * 70)
        
        # 验证数据
        print("\n📊 数据验证:")
        print("-" * 70)
        
        # 获取一个批次
        for batch_x, batch_y in train_loader:
            print(f"✓ 训练批次形状:")
            print(f"  - 输入 (X): {batch_x.shape}")
            print(f"  - 目标 (y): {batch_y.shape}")
            print(f"  - 数据类型: {batch_x.dtype}")
            break
        
        for batch_x, batch_y in val_loader:
            print(f"\n✓ 验证批次形状:")
            print(f"  - 输入 (X): {batch_x.shape}")
            print(f"  - 目标 (y): {batch_y.shape}")
            break
        
        for batch_x, batch_y in test_loader:
            print(f"\n✓ 测试批次形状:")
            print(f"  - 输入 (X): {batch_x.shape}")
            print(f"  - 目标 (y): {batch_y.shape}")
            break
        
        # 测试反归一化
        print("\n" + "-" * 70)
        print("🔄 测试反归一化功能:")
        print("-" * 70)
        
        import numpy as np
        test_data = np.array([[0.5], [0.6], [0.7]])
        
        # 获取第一个特征的scaler
        feature_names = list(preprocessor.scalers.keys())
        if feature_names:
            first_feature = feature_names[0]
            original = preprocessor.inverse_transform(test_data, first_feature)
            print(f"✓ 归一化数据: {test_data.flatten()}")
            print(f"✓ 反归一化后: {original}")
        
        print("\n" + "=" * 70)
        print("🎉 所有测试通过！")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ 测试失败: {str(e)}")
        print("=" * 70 + "\n")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """测试各个组件"""
    
    print("\n" + "=" * 70)
    print("🧪 测试各个组件")
    print("=" * 70 + "\n")
    
    data_dir = project_root / 'data'
    
    try:
        # 测试DataLoader
        print("1️⃣ 测试 DataLoader")
        print("-" * 70)
        from src.data import DataLoader
        
        loader = DataLoader(str(data_dir))
        data_dict = loader.load_all_data()
        
        print(f"✓ 加载了 {len(data_dict)} 个数据源:")
        for name, df in data_dict.items():
            print(f"  - {name}: {df.shape}")
        
        merged = loader.merge_data_sources(data_dict)
        print(f"✓ 合并后数据形状: {merged.shape}")
        
        # 测试DataPreprocessor
        print("\n2️⃣ 测试 DataPreprocessor")
        print("-" * 70)
        from src.data import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # 测试缺失值处理
        processed = preprocessor.handle_missing_values(merged)
        print(f"✓ 缺失值处理完成，剩余缺失值: {processed.isnull().sum().sum()}")
        
        # 测试异常值检测
        outliers = preprocessor.detect_outliers(processed)
        print(f"✓ 检测到 {outliers.sum().sum()} 个异常值")
        
        # 测试归一化
        normalized = preprocessor.normalize(processed)
        print(f"✓ 归一化完成，数据范围: [{normalized.min().min():.3f}, {normalized.max().max():.3f}]")
        
        # 测试时间窗口创建
        import numpy as np
        data_array = normalized.values
        X, y = preprocessor.create_time_windows(data_array, window_size=21, horizon=7)
        print(f"✓ 时间窗口创建完成: X={X.shape}, y={y.shape}")
        
        # 测试数据划分
        train, val, test = preprocessor.temporal_train_test_split(X)
        print(f"✓ 数据划分完成: train={len(train)}, val={len(val)}, test={len(test)}")
        
        # 测试Dataset
        print("\n3️⃣ 测试 EpidemicDataset")
        print("-" * 70)
        from src.data import EpidemicDataset
        
        dataset = EpidemicDataset(X[:100], y[:100])
        print(f"✓ 数据集大小: {len(dataset)}")
        print(f"✓ 特征维度: {dataset.get_feature_dim()}")
        print(f"✓ 窗口大小: {dataset.get_window_size()}")
        
        sample_x, sample_y = dataset[0]
        print(f"✓ 样本形状: X={sample_x.shape}, y={sample_y.shape}")
        
        print("\n" + "=" * 70)
        print("✅ 所有组件测试通过！")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ 组件测试失败: {str(e)}")
        print("=" * 70 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 运行测试
    success1 = test_individual_components()
    success2 = test_data_pipeline()
    
    if success1 and success2:
        print("\n🎊 恭喜！数据预处理模块已完全实现并通过测试！\n")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查错误信息\n")
        sys.exit(1)
