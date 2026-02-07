"""
Configuration utilities.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class DataConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    processed_csv: str = ""

    window_size: int = 21
    prediction_horizon: int = 7
    stride: int = 1

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    scaler_type: str = "standard"
    handle_missing: str = "interpolate"
    per_state_normalize: bool = False
    target_log1p: bool = False
    inverse_transform: bool = False
    state_column: str = "State"

    target_column: str = "new_cases"
    feature_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    num_variables: int = 10
    input_size: int = 1
    tcn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    tcn_kernel_size: int = 3

    attention_embed_dim: int = 128
    attention_num_heads: int = 8
    attention_dropout: float = 0.1

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True

    output_size: int = 1
    prediction_horizon: int = 7

    dropout: float = 0.2

    use_state_embedding: bool = False
    num_states: int = 0
    state_embed_dim: int = 16


@dataclass
class AttentionConfig:
    type: str = "spatiotemporal"
    num_heads: int = 4
    dropout: float = 0.1
    stochastic_dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    optimizer: str = "adamw"

    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    use_pretrain: bool = True
    pretrain_epochs: int = 20
    finetune_lr_ratio: float = 0.1

    gradient_clip_val: float = 1.0

    device: str = "cuda"
    num_workers: int = 4

    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    log_interval: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    seed: int = 42
    experiment_name: str = "attention_mtcn_lstm"

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'attention': asdict(self.attention),
            'seed': self.seed,
            'experiment_name': self.experiment_name,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        attention_config = AttentionConfig(**config_dict.get('attention', {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            attention=attention_config,
            seed=config_dict.get('seed', 42),
            experiment_name=config_dict.get('experiment_name', 'attention_mtcn_lstm'),
        )


def load_config(config_path: Union[str, Path]) -> Config:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return Config.from_dict(config_dict)


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_default_config() -> Config:
    return Config()
