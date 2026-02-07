import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.utils.visualization import Visualizer


def test_plot_training_history_returns_figure():
    visualizer = Visualizer()
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "learning_rate": [0.01, 0.005, 0.001],
    }

    fig = visualizer.plot_training_history(history, title="History")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    loss_ax = fig.axes[0]
    lr_ax = fig.axes[1]
    assert len(loss_ax.lines) >= 2
    assert len(lr_ax.lines) == 1


def test_plot_correlation_matrix_returns_figure():
    visualizer = Visualizer()
    data = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.5, 4.0],
        [2.0, 4.0, 6.0],
    ])
    features = ["a", "b", "c"]

    fig = visualizer.plot_correlation_matrix(data, features, title="Corr")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_time_series_returns_figure():
    visualizer = Visualizer()
    series = np.array([1, 2, 3, 2, 1])
    dates = ["d1", "d2", "d3", "d4", "d5"]

    fig = visualizer.plot_time_series(series, dates=dates, title="Series")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Series"


def test_plot_multi_series_returns_figure():
    visualizer = Visualizer()
    series_dict = {
        "a": np.array([1, 2, 3]),
        "b": np.array([2, 3, 4]),
    }

    fig = visualizer.plot_multi_series(series_dict, title="Multi")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2


def test_plot_residuals_multistep_returns_figure():
    visualizer = Visualizer()
    actual = np.random.randn(100, 4)  # multi-step forecasts
    predicted = actual + 0.1 * np.random.randn(100, 4)

    fig = visualizer.plot_residuals(actual, predicted, title="Residuals")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
