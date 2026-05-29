import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import mae, rmse, r2, evaluate_metrics


class TestMetrics:
    """Unit tests for metrics functions."""

    def test_mae_known_values(self):
        """Test MAE with known values."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        expected_mae = 0.0
        assert mae(y_true, y_pred) == expected_mae

    def test_mae_simple_case(self):
        """Test MAE with simple case."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        expected_mae = (0.5 + 0.5 + 0.0 + 1.0) / 4  # 0.5
        assert np.isclose(mae(y_true, y_pred), expected_mae)

    def test_rmse_known_values(self):
        """Test RMSE with known values."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        expected_mse = (0.5**2 + 0.5**2 + 0.0**2 + 1.0**2) / 4
        expected_rmse = np.sqrt(expected_mse)
        assert np.isclose(rmse(y_true, y_pred), expected_rmse)

    def test_rmse_perfect_prediction(self):
        """Test RMSE when predictions are perfect."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_r2_perfect_prediction(self):
        """Test R² when predictions are perfect."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2(y_true, y_pred) == 1.0

    def test_r2_constant_prediction(self):
        """Test R² when predicting mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # mean value
        assert np.isclose(r2(y_true, y_pred), 0.0)

    def test_evaluate_metrics_returns_dict(self):
        """Test evaluate_metrics returns dictionary with correct keys."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])
        results = evaluate_metrics(y_true, y_pred)

        assert isinstance(results, dict)
        assert 'MAE' in results
        assert 'RMSE' in results
        assert 'R2' in results
        assert all(isinstance(v, (int, float)) for v in results.values())

    def test_metrics_consistency(self):
        """Test that individual metrics match evaluate_metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        r2_val = r2(y_true, y_pred)

        results = evaluate_metrics(y_true, y_pred)

        assert np.isclose(mae_val, results['MAE'])
        assert np.isclose(rmse_val, results['RMSE'])
        assert np.isclose(r2_val, results['R2'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
