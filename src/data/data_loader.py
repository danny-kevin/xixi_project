"""
数据加载器模块
Data Loader Module

负责从各种数据源加载原始数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class DataLoader:
    """
    多源异构数据加载器
    
    支持加载四类数据:
    - 疫情数据: 每日新增确诊、死亡、康复病例
    - 人口流动数据: 手机定位、交通枢纽人流指数
    - 环境数据: 温度、湿度、紫外线强度
    - 干预政策数据: 封城等级、社交距离、疫苗接种率
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据根目录路径
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
    def load_epidemic_data(self, filename: str) -> pd.DataFrame:
        """
        加载疫情数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            包含疫情数据的DataFrame
        """
        filepath = self.raw_dir / filename
        
        # 支持CSV和Excel格式
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        # 确保包含必要的列
        required_cols = ['date', 'new_cases']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"疫情数据缺少必要列: {missing_cols}")
        
        # 设置日期为索引
        df = df.set_index('date').sort_index()
        
        return df
    
    def load_mobility_data(self, filename: str) -> pd.DataFrame:
        """
        加载人口流动数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            包含人口流动数据的DataFrame
        """
        filepath = self.raw_dir / filename
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        # 设置日期为索引
        df = df.set_index('date').sort_index()
        
        return df
    
    def load_environmental_data(self, filename: str) -> pd.DataFrame:
        """
        加载环境数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            包含环境数据的DataFrame
        """
        filepath = self.raw_dir / filename
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        # 设置日期为索引
        df = df.set_index('date').sort_index()
        
        return df
    
    def load_intervention_data(self, filename: str) -> pd.DataFrame:
        """
        加载干预政策数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            包含干预政策数据的DataFrame
        """
        filepath = self.raw_dir / filename
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        # 设置日期为索引
        df = df.set_index('date').sort_index()
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有类型的数据
        
        Returns:
            包含所有数据类型的字典
        """
        data_dict = {}
        
        # 尝试加载各类数据（如果文件存在）
        try:
            data_dict['epidemic'] = self.load_epidemic_data('epidemic.csv')
        except FileNotFoundError:
            print("警告: 未找到疫情数据文件")
        
        try:
            data_dict['mobility'] = self.load_mobility_data('mobility.csv')
        except FileNotFoundError:
            print("警告: 未找到人口流动数据文件")
        
        try:
            data_dict['environmental'] = self.load_environmental_data('environmental.csv')
        except FileNotFoundError:
            print("警告: 未找到环境数据文件")
        
        try:
            data_dict['intervention'] = self.load_intervention_data('intervention.csv')
        except FileNotFoundError:
            print("警告: 未找到干预政策数据文件")
        
        return data_dict
    
    def merge_data_sources(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        on: str = 'date'
    ) -> pd.DataFrame:
        """
        合并多源数据
        
        Args:
            data_dict: 数据字典
            on: 合并的键列名（默认为索引）
            
        Returns:
            合并后的DataFrame
        """
        if not data_dict:
            raise ValueError("数据字典为空，无法合并")
        
        # 从第一个数据源开始
        merged_df = None
        
        for source_name, df in data_dict.items():
            if merged_df is None:
                merged_df = df.copy()
                # 添加数据源前缀
                merged_df.columns = [f"{source_name}_{col}" for col in merged_df.columns]
            else:
                # 添加数据源前缀
                df_renamed = df.copy()
                df_renamed.columns = [f"{source_name}_{col}" for col in df_renamed.columns]
                
                # 按日期索引合并（外连接，保留所有日期）
                merged_df = merged_df.join(df_renamed, how='outer')
        
        # 按日期排序
        merged_df = merged_df.sort_index()
        
        return merged_df
