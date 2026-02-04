"""
生成示例数据
Generate Sample Data

用于测试数据预处理管道的示例数据生成脚本
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def generate_epidemic_data(
    start_date: str = '2020-01-01',
    n_days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟疫情数据
    
    Args:
        start_date: 起始日期
        n_days: 天数
        seed: 随机种子
        
    Returns:
        疫情数据DataFrame
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 模拟疫情曲线（使用正弦波 + 噪声）
    t = np.arange(n_days)
    base_cases = 100 + 50 * np.sin(2 * np.pi * t / 30) + np.random.randn(n_days) * 10
    base_cases = np.maximum(base_cases, 0)  # 确保非负
    
    data = {
        'date': dates,
        'new_cases': base_cases,
        'new_deaths': base_cases * 0.02 + np.random.randn(n_days) * 0.5,
        'new_recovered': base_cases * 0.8 + np.random.randn(n_days) * 5,
        'cumulative_cases': np.cumsum(base_cases),
        'cumulative_deaths': np.cumsum(base_cases * 0.02),
        'active_cases': base_cases * 10 + np.random.randn(n_days) * 20
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值（5%）
    mask = np.random.rand(n_days) < 0.05
    df.loc[mask, 'new_cases'] = np.nan
    
    return df


def generate_mobility_data(
    start_date: str = '2020-01-01',
    n_days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟人口流动数据
    
    Args:
        start_date: 起始日期
        n_days: 天数
        seed: 随机种子
        
    Returns:
        人口流动数据DataFrame
    """
    np.random.seed(seed + 1)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 模拟流动性数据（工作日高，周末低）
    t = np.arange(n_days)
    weekday_effect = np.array([0.8 if (i % 7) < 5 else 0.5 for i in range(n_days)])
    
    data = {
        'date': dates,
        'intra_city_flow': 100 * weekday_effect + np.random.randn(n_days) * 10,
        'inter_city_flow': 80 * weekday_effect + np.random.randn(n_days) * 8,
        'public_transport': 90 * weekday_effect + np.random.randn(n_days) * 9,
        'retail_mobility': 70 + 20 * np.sin(2 * np.pi * t / 7) + np.random.randn(n_days) * 5,
        'workplace_mobility': 85 * weekday_effect + np.random.randn(n_days) * 7
    }
    
    df = pd.DataFrame(data)
    return df


def generate_environmental_data(
    start_date: str = '2020-01-01',
    n_days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟环境数据
    
    Args:
        start_date: 起始日期
        n_days: 天数
        seed: 随机种子
        
    Returns:
        环境数据DataFrame
    """
    np.random.seed(seed + 2)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 模拟季节性温度变化
    t = np.arange(n_days)
    
    data = {
        'date': dates,
        'temperature': 15 + 10 * np.sin(2 * np.pi * t / 365) + np.random.randn(n_days) * 3,
        'humidity': 60 + 15 * np.sin(2 * np.pi * t / 365 + np.pi/2) + np.random.randn(n_days) * 5,
        'uv_index': 5 + 3 * np.sin(2 * np.pi * t / 365) + np.random.randn(n_days) * 1,
        'precipitation': np.maximum(0, 5 + 3 * np.sin(2 * np.pi * t / 365 + np.pi) + np.random.randn(n_days) * 2),
        'wind_speed': 3 + 2 * np.random.randn(n_days)
    }
    
    df = pd.DataFrame(data)
    return df


def generate_intervention_data(
    start_date: str = '2020-01-01',
    n_days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟干预政策数据
    
    Args:
        start_date: 起始日期
        n_days: 天数
        seed: 随机种子
        
    Returns:
        干预政策数据DataFrame
    """
    np.random.seed(seed + 3)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 模拟政策变化（阶段性）
    lockdown_level = np.zeros(n_days)
    lockdown_level[60:120] = 3  # 第一波封锁
    lockdown_level[200:250] = 2  # 第二波封锁
    
    social_distance = np.zeros(n_days)
    social_distance[60:180] = 2
    social_distance[180:] = 1
    
    mask_mandate = np.zeros(n_days)
    mask_mandate[90:] = 1
    
    # 疫苗接种率逐步上升
    vaccination_rate = np.zeros(n_days)
    vaccination_rate[180:] = np.linspace(0, 60, n_days - 180)
    
    data = {
        'date': dates,
        'lockdown_level': lockdown_level,
        'social_distance': social_distance,
        'mask_mandate': mask_mandate,
        'vaccination_rate': vaccination_rate,
        'testing_rate': 5 + np.random.randn(n_days) * 1
    }
    
    df = pd.DataFrame(data)
    return df


def generate_all_sample_data(
    output_dir: str = 'data/raw',
    start_date: str = '2020-01-01',
    n_days: int = 365,
    seed: int = 42
):
    """
    生成所有类型的示例数据并保存
    
    Args:
        output_dir: 输出目录
        start_date: 起始日期
        n_days: 天数
        seed: 随机种子
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🎲 生成示例数据")
    print("=" * 60)
    
    # 生成疫情数据
    print("📊 生成疫情数据...")
    epidemic_df = generate_epidemic_data(start_date, n_days, seed)
    epidemic_path = output_path / 'epidemic.csv'
    epidemic_df.to_csv(epidemic_path, index=False)
    print(f"   ✅ 保存到: {epidemic_path}")
    print(f"   形状: {epidemic_df.shape}")
    
    # 生成人口流动数据
    print("🚶 生成人口流动数据...")
    mobility_df = generate_mobility_data(start_date, n_days, seed)
    mobility_path = output_path / 'mobility.csv'
    mobility_df.to_csv(mobility_path, index=False)
    print(f"   ✅ 保存到: {mobility_path}")
    print(f"   形状: {mobility_df.shape}")
    
    # 生成环境数据
    print("🌡️ 生成环境数据...")
    environmental_df = generate_environmental_data(start_date, n_days, seed)
    environmental_path = output_path / 'environmental.csv'
    environmental_df.to_csv(environmental_path, index=False)
    print(f"   ✅ 保存到: {environmental_path}")
    print(f"   形状: {environmental_df.shape}")
    
    # 生成干预政策数据
    print("📋 生成干预政策数据...")
    intervention_df = generate_intervention_data(start_date, n_days, seed)
    intervention_path = output_path / 'intervention.csv'
    intervention_df.to_csv(intervention_path, index=False)
    print(f"   ✅ 保存到: {intervention_path}")
    print(f"   形状: {intervention_df.shape}")
    
    print("=" * 60)
    print("✅ 所有示例数据生成完成！")
    print("=" * 60)
    
    return {
        'epidemic': epidemic_df,
        'mobility': mobility_df,
        'environmental': environmental_df,
        'intervention': intervention_df
    }


if __name__ == '__main__':
    # 生成示例数据
    generate_all_sample_data(
        output_dir='data/raw',
        start_date='2020-01-01',
        n_days=365,
        seed=42
    )
