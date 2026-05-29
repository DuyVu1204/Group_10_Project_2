"""
test_models.py – Unit tests cho model_comparison.py (Checklist §6.7).

Tests:
    - fit_ols_full chạy được với dữ liệu nhỏ.
    - fit_ridge_cv trả về best_lambda nằm trong lambda grid.
    - fit_lasso_cv trả về best_lambda nằm trong lambda grid.
    - evaluate_models trả về đủ cột Model, MAE, RMSE, R2.
    - Model predict đúng số lượng dòng.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_comparison import (
    fit_ols_full,
    fit_ols_selected,
    fit_ridge_cv,
    fit_lasso_cv,
    evaluate_models,
    get_best_model,
    LinearModel,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_data():
    """Tạo bộ dữ liệu tổng hợp nhỏ, dễ kiểm chứng."""
    np.random.seed(42)
    n, p = 80, 4
    X = np.random.randn(n, p)
    true_coef = np.array([3.0, -1.5, 0.0, 2.0])
    y = X @ true_coef + 5.0 + np.random.randn(n) * 0.5
    return X, y


@pytest.fixture
def synth_train_test(synth_data):
    """Split 80 → 60 train / 20 test."""
    X, y = synth_data
    return X[:60], X[60:], y[:60], y[60:]


@pytest.fixture
def lambda_grid():
    return np.logspace(-3, 3, 20)


# ---------------------------------------------------------------------------
#  Tests – OLS Full
# ---------------------------------------------------------------------------

class TestOLSFull:

    def test_fit_ols_full_returns_linear_model(self, synth_train_test):
        X_tr, _, y_tr, _ = synth_train_test
        model = fit_ols_full(X_tr, y_tr)
        assert isinstance(model, LinearModel)

    def test_ols_full_coef_shape(self, synth_train_test):
        X_tr, _, y_tr, _ = synth_train_test
        model = fit_ols_full(X_tr, y_tr)
        assert model.coef_.shape == (X_tr.shape[1],)

    def test_ols_full_predict_shape(self, synth_train_test):
        X_tr, X_te, y_tr, _ = synth_train_test
        model = fit_ols_full(X_tr, y_tr)
        y_pred = model.predict(X_te)
        assert y_pred.shape == (X_te.shape[0],)

    def test_ols_full_near_true_coef(self, synth_train_test):
        """Với dữ liệu ít nhiễu, beta_hat phải gần beta_true."""
        X_tr, _, y_tr, _ = synth_train_test
        model = fit_ols_full(X_tr, y_tr)
        true_coef = np.array([3.0, -1.5, 0.0, 2.0])
        assert np.allclose(model.coef_, true_coef, atol=0.5)


# ---------------------------------------------------------------------------
#  Tests – OLS Selected
# ---------------------------------------------------------------------------

class TestOLSSelected:

    def test_fit_ols_selected_with_dataframe(self, synth_train_test):
        X_tr, _, y_tr, _ = synth_train_test
        df = pd.DataFrame(X_tr, columns=["a", "b", "c", "d"])
        model = fit_ols_selected(df, y_tr, ["a", "b"])
        assert model.coef_.shape == (2,)


# ---------------------------------------------------------------------------
#  Tests – Ridge CV
# ---------------------------------------------------------------------------

class TestRidgeCV:

    def test_ridge_cv_returns_tuple(self, synth_train_test, lambda_grid):
        X_tr, _, y_tr, _ = synth_train_test
        result = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        assert len(result) == 3

    def test_ridge_best_lambda_in_grid(self, synth_train_test, lambda_grid):
        X_tr, _, y_tr, _ = synth_train_test
        _, best_lam, _ = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        assert best_lam in lambda_grid

    def test_ridge_cv_results_df_columns(self, synth_train_test, lambda_grid):
        X_tr, _, y_tr, _ = synth_train_test
        _, _, cv_df = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        assert "lambda" in cv_df.columns
        assert "mean_cv_rmse" in cv_df.columns
        assert "std_cv_rmse" in cv_df.columns

    def test_ridge_cv_results_df_length(self, synth_train_test, lambda_grid):
        X_tr, _, y_tr, _ = synth_train_test
        _, _, cv_df = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        assert len(cv_df) == len(lambda_grid)

    def test_ridge_predict_shape(self, synth_train_test, lambda_grid):
        X_tr, X_te, y_tr, _ = synth_train_test
        model, _, _ = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        y_pred = model.predict(X_te)
        assert y_pred.shape == (X_te.shape[0],)


# ---------------------------------------------------------------------------
#  Tests – Lasso CV
# ---------------------------------------------------------------------------

class TestLassoCV:

    def test_lasso_cv_returns_tuple(self, synth_train_test, lambda_grid):
        X_tr, _, y_tr, _ = synth_train_test
        result = fit_lasso_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        assert len(result) == 3

    def test_lasso_best_lambda_in_grid(self, synth_train_test, lambda_grid):
        X_tr, _, y_tr, _ = synth_train_test
        _, best_lam, _ = fit_lasso_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        assert best_lam in lambda_grid

    def test_lasso_predict_shape(self, synth_train_test, lambda_grid):
        X_tr, X_te, y_tr, _ = synth_train_test
        model, _, _ = fit_lasso_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        y_pred = model.predict(X_te)
        assert y_pred.shape == (X_te.shape[0],)


# ---------------------------------------------------------------------------
#  Tests – evaluate_models
# ---------------------------------------------------------------------------

class TestEvaluateModels:

    def test_evaluate_returns_dataframe(self, synth_train_test):
        X_tr, X_te, y_tr, y_te = synth_train_test
        m1 = fit_ols_full(X_tr, y_tr)
        models = {"OLS Full": m1}
        result = evaluate_models(models, X_te, y_te)
        assert isinstance(result, pd.DataFrame)

    def test_evaluate_has_required_columns(self, synth_train_test):
        X_tr, X_te, y_tr, y_te = synth_train_test
        m1 = fit_ols_full(X_tr, y_tr)
        models = {"OLS Full": m1}
        result = evaluate_models(models, X_te, y_te)
        for col in ["Model", "MAE", "RMSE", "R2"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_evaluate_multiple_models(self, synth_train_test, lambda_grid):
        X_tr, X_te, y_tr, y_te = synth_train_test
        m_ols = fit_ols_full(X_tr, y_tr)
        m_ridge, _, _ = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        models = {"OLS Full": m_ols, "Ridge CV": m_ridge}
        result = evaluate_models(models, X_te, y_te)
        assert len(result) == 2

    def test_evaluate_r2_positive(self, synth_train_test):
        """Với dữ liệu dễ, R² phải > 0."""
        X_tr, X_te, y_tr, y_te = synth_train_test
        m = fit_ols_full(X_tr, y_tr)
        result = evaluate_models({"OLS": m}, X_te, y_te)
        assert result.iloc[0]["R2"] > 0


# ---------------------------------------------------------------------------
#  Tests – get_best_model
# ---------------------------------------------------------------------------

class TestGetBestModel:

    def test_get_best_model_returns_string(self, synth_train_test, lambda_grid):
        X_tr, X_te, y_tr, y_te = synth_train_test
        m_ols = fit_ols_full(X_tr, y_tr)
        m_ridge, _, _ = fit_ridge_cv(X_tr, y_tr, lambda_grid, n_splits=3)
        models = {"OLS Full": m_ols, "Ridge CV": m_ridge}
        results = evaluate_models(models, X_te, y_te)
        best = get_best_model(results, "RMSE")
        assert isinstance(best, str)
        assert best in ["OLS Full", "Ridge CV"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
