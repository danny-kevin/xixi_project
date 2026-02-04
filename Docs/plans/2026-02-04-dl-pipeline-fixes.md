# DL Pipeline Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make train/eval/predict/experiment run end-to-end on `dataset_US_final.csv` with state one-hot features, per-state windows, ASCII-safe logging, and fixed metrics/visualization paths.

**Architecture:** Use `src/data/pipeline.py` as the unified data path for all modes. Add a single-file CSV path with `Date` parsing, `State` one-hot encoding, per-state windowing, and optional normalization skip. Align model input sizes from config, fix metrics API usage, and implement required visualization stubs used by experiment mode. Replace non-ASCII console output to avoid GBK errors.

**Tech Stack:** Python, PyTorch, pandas, numpy, scikit-learn, pytest.

### Task 1: Add failing tests for new pipeline behavior

**Files:**
- Modify: `tests/test_data_pipeline.py`
- Create: `tests/test_us_single_file_pipeline.py`

**Step 1: Write failing test for per-state windowing with one-hot**

```python
def test_us_single_file_pipeline_windows_per_state(tmp_path):
    # load dataset_US_final.csv and run pipeline single-file path
    # assert no window crosses state boundary and State one-hot columns exist
```

**Step 2: Run test to verify it fails**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe -m pytest tests/test_us_single_file_pipeline.py -q`
Expected: FAIL because pipeline lacks single-file + state one-hot logic.

**Step 3: Update scripts/test_data_pipeline.py to return None (assert instead of return)**

```python
def test_data_pipeline():
    ...
    assert True
```

**Step 4: Run test to verify it fails or still warns**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe -m pytest scripts/test_data_pipeline.py -q`
Expected: FAIL until pipeline is updated to accept the new CSV format.

**Step 5: Commit**

```bash
git add tests/test_us_single_file_pipeline.py tests/test_data_pipeline.py
git commit -m "test: add pipeline coverage for US single-file data"
```

### Task 2: Implement single-file pipeline with state one-hot + per-state windows

**Files:**
- Modify: `src/data/pipeline.py`
- Modify: `src/data/preprocessor.py`

**Step 1: Implement CSV single-file load path**

```python
def load_data(self):
    # detect dataset_US_final.csv
    # parse Date, set index, sort
```

**Step 2: Add state one-hot encoding**

```python
df_processed = pd.get_dummies(df_processed, columns=["State"], prefix="State")
```

**Step 3: Add per-state windowing**

```python
for state, group in df.groupby("State"):
    X_state, y_state = self.preprocessor.create_time_windows(...)
    collect and concat
```

**Step 4: Allow skip normalization**

```python
if not normalize:
    df_processed = df_processed
```

**Step 5: Update DataPreprocessor.create_time_windows to accept target column**

```python
def create_time_windows(..., target_col_idx: int = 0):
    y.append(data[i + window_size, target_col_idx])
```

**Step 6: Run tests**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe -m pytest tests/test_us_single_file_pipeline.py -q`
Expected: PASS

**Step 7: Commit**

```bash
git add src/data/pipeline.py src/data/preprocessor.py
git commit -m "feat: add single-file US pipeline with state one-hot and per-state windows"
```

### Task 3: Wire train/eval/predict/experiment to pipeline and fix metrics usage

**Files:**
- Modify: `train.py`
- Modify: `main.py`
- Modify: `run_experiment.py`
- Modify: `src/evaluation/metrics.py`

**Step 1: Replace USCovidDataLoader usage with DataPipeline**

```python
from src.data.pipeline import DataPipeline
pipeline = DataPipeline(..., use_single_file=True, normalize=False)
```

**Step 2: Import torch in main.py and fix metrics API call**

```python
import torch
metrics = RegressionMetrics.compute_all(...)
```

**Step 3: Update config usage to set num_variables and feature list from pipeline outputs**

```python
config.model.num_variables = num_features
config.data.feature_columns = feature_names
```

**Step 4: Run pytest**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe -m pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add train.py main.py run_experiment.py src/evaluation/metrics.py
git commit -m "fix: unify pipeline in train/eval/predict/experiment and correct metrics usage"
```

### Task 4: Implement visualization stubs required by experiment

**Files:**
- Modify: `src/utils/visualization.py`

**Step 1: Implement plot_training_history**

```python
def plot_training_history(...):
    # plot train/val loss and lr
```

**Step 2: Implement plot_correlation_matrix / plot_time_series / plot_multi_series**

```python
def plot_correlation_matrix(...):
    # seaborn heatmap
```

**Step 3: Run experiment smoke test**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe main.py --mode experiment --config configs/default_config.yaml --device cpu`
Expected: RUN completes without NotImplementedError.

**Step 4: Commit**

```bash
git add src/utils/visualization.py
git commit -m "feat: implement visualization helpers for experiment mode"
```

### Task 5: Remove non-ASCII console output for GBK safety

**Files:**
- Modify: `verify_framework.py`
- Modify: `load_us_data.py`
- Modify: `scripts/test_data_pipeline.py`
- Modify: `src/data/pipeline.py`

**Step 1: Replace emoji / non-ASCII prints with ASCII**

```python
print("[OK] ...")
print("[WARN] ...")
```

**Step 2: Run smoke tests**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe verify_framework.py`
Expected: PASS without UnicodeEncodeError

**Step 3: Commit**

```bash
git add verify_framework.py load_us_data.py scripts/test_data_pipeline.py src/data/pipeline.py
git commit -m "fix: use ASCII-only console output for Windows GBK"
```

### Task 6: End-to-end validation for all modes

**Files:**
- Modify: `scripts/smoke_all_modes.py` (create helper)

**Step 1: Add a smoke script**

```python
subprocess.run([python, "main.py", "--mode", "train", "--epochs", "1", "--device", "cpu"])
subprocess.run([python, "main.py", "--mode", "eval", "--checkpoint", "checkpoints/best_model.pth", "--device", "cpu"])
subprocess.run([python, "main.py", "--mode", "predict", "--checkpoint", "checkpoints/best_model.pth", "--device", "cpu"])
subprocess.run([python, "main.py", "--mode", "experiment", "--device", "cpu"])
```

**Step 2: Run smoke script**

Run: `F:\01_Learn_Work\Code\xixi_project\.conda\python.exe scripts/smoke_all_modes.py`
Expected: All subprocesses return 0.

**Step 3: Commit**

```bash
git add scripts/smoke_all_modes.py
git commit -m "test: add smoke script for train/eval/predict/experiment"
```
