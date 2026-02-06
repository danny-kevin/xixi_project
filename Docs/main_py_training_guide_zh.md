# 使用 main.py 训练与参数配置说明（单文件数据）

本项目推荐使用 `main.py` 作为统一入口运行 `train/eval/predict/experiment`。当你用单文件 `data/raw/dataset_US_final.csv` 训练时，实际的数据与特征处理走 `src/data/pipeline.py` 的单文件路径，并在训练阶段由 `train.py` 负责创建模型、优化器、学习率调度器以及早停逻辑。

## 1. 最小可运行命令

单文件训练（CPU 示例）：

```powershell
F:\01_Learn_Work\Code\xixi_project\.conda\python.exe main.py --mode train --config configs/default_config.yaml --device cpu
```

常用覆盖参数：

```powershell
F:\01_Learn_Work\Code\xixi_project\.conda\python.exe main.py --mode train --config configs/default_config.yaml --device cpu --epochs 50 --batch-size 32 --learning-rate 0.0005
```

说明：
- `--config` 指向你要使用的 YAML 配置。
- `--epochs/--batch-size/--learning-rate/--device/--seed` 会覆盖配置文件里的对应值。
- 输出目录（日志等）由 `--output-dir` 控制，默认 `results`。

## 2. 早停（Early Stopping）怎么生效

早停逻辑在 `src/training/trainer.py` 的 `Trainer.finetune()` 中实现：
- 每个 epoch 计算 `val_loss`。
- 如果在连续 `patience` 次验证中，`val_loss` 都没有比历史最佳值改善至少 `min_delta`，则触发早停并提前结束训练。

你需要在配置文件里设置下面两个字段（位于 `training` 下）：
- `early_stopping_patience`: 容忍多少个 epoch 没有改进。
- `early_stopping_min_delta`: 认为“有改进”的最小阈值。

例如（`configs/default_config.yaml`）：

```yaml
training:
  early_stopping_patience: 15
  early_stopping_min_delta: 0.0001
```

重要提示（和你的训练行为直接相关）：
- 早停只作用在 finetune 阶段。
- 如果你开启了预训练：`training.use_pretrain: true`，则预训练阶段会固定跑满 `training.pretrain_epochs`，不会早停；之后进入 finetune 才会早停。

## 3. 其他常用可调参数（建议优先改 YAML）

推荐工作流：
1. 复制 `configs/default_config.yaml` 为新文件，例如 `configs/run1.yaml`
2. 只在 `run1.yaml` 修改参数
3. 训练时用 `--config configs/run1.yaml`

### 3.1 数据与预测设置（`data`）
- `window_size`: 输入历史窗口长度。
- `prediction_horizon`: 预测步长/范围。
- `target_column`: 要预测的目标列名（例如 `Confirmed`）。
- `train_ratio/val_ratio/test_ratio`: 时序切分比例。

### 3.2 训练设置（`training`）
以下字段通常会影响训练行为：
- `epochs`
- `batch_size`
- `learning_rate`
- `weight_decay`
- `warmup_epochs`
- `min_lr`
- `gradient_clip_val`
- `use_pretrain`
- `pretrain_epochs`
- `finetune_lr_ratio`
- `checkpoint_dir`
- `num_workers`
- `early_stopping_patience`
- `early_stopping_min_delta`

### 3.3 模型结构（`model`）
常见需要调的字段：
- `tcn_channels`
- `tcn_kernel_size`
- `attention_embed_dim/attention_num_heads/attention_dropout`
- `lstm_hidden_size/lstm_num_layers/lstm_bidirectional`
- `dropout`

## 4. main.py 支持的命令行覆盖项（当前实现）

`main.py` 当前支持这些 CLI 参数覆盖：
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--device`
- `--seed`
- `--output-dir`
- `--checkpoint`
- `--log-level`
- `--use-wandb`

如果你希望把更多训练字段（比如 `early_stopping_patience/min_delta`、`weight_decay`、`warmup_epochs` 等）也做成 CLI 参数，需要对 `main.py` 和/或 `train.py` 做小幅改造。

## 5. 当前训练链路里“被硬编码/会被覆盖”的点（避免踩坑）

这些点会影响你“改了 YAML 但没生效”的排查方向：
- `train.py` 会用 pipeline 自动生成的特征名覆盖 `config.data.feature_columns`，并重设 `config.model.num_variables`。
- 单文件训练路径里 `normalize=False`（也就是不会做归一化）；如果你希望归一化，需要改 `train.py` 中构建 `DataPipeline(...)` 的参数。

## 6. eval/predict 的运行方式

评估（需要 `--checkpoint`）：

```powershell
F:\01_Learn_Work\Code\xixi_project\.conda\python.exe main.py --mode eval --config configs/default_config.yaml --device cpu --checkpoint checkpoints/best_model.pth
```

预测初始化（需要 `--checkpoint`）：

```powershell
F:\01_Learn_Work\Code\xixi_project\.conda\python.exe main.py --mode predict --config configs/default_config.yaml --device cpu --checkpoint checkpoints/best_model.pth
```

## 7. 端到端自检（推荐）

如果你只是想快速确认四个模式都能跑通：

```powershell
F:\01_Learn_Work\Code\xixi_project\.conda\python.exe scripts/smoke_all_modes.py
```

