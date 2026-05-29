
from __future__ import annotations

from typing import Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def _as_1d_array(y: ArrayLike) -> np.ndarray:
    array = np.asarray(y, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError("y must be a 1D array or a column vector")
    return array


def _as_2d_array(X: ArrayLike) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("X must be a 1D or 2D array")
    return array


def add_intercept(X: ArrayLike) -> np.ndarray:
    X = _as_2d_array(X)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.column_stack((ones, X))


def ols_fit(X: ArrayLike, y: ArrayLike, add_intercept_column: bool = True) -> dict:

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    design = add_intercept(X) if add_intercept_column else X
    xtx = design.T @ design
    xty = design.T @ y

    try:
        beta_hat = np.linalg.solve(xtx, xty)
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("X'X is singular or ill-conditioned") from exc

    y_hat = design @ beta_hat
    residuals = y - y_hat
    n_samples, n_params = design.shape
    degrees_of_freedom = n_samples - n_params
    if degrees_of_freedom <= 0:
        raise ValueError("Not enough degrees of freedom to estimate sigma^2")
    rss = float(residuals.T @ residuals)
    sigma2_hat = rss / degrees_of_freedom

    return {
        "beta_hat": beta_hat,
        "sigma2_hat": sigma2_hat,
        "residuals": residuals,
        "y_hat": y_hat,
        "design_matrix": design,
        "xtx": xtx,
        "xtx_inv": xtx_inv,
        "n_samples": n_samples,
        "n_params": n_params,
        "df_resid": degrees_of_freedom,
    }


def hat_matrix(X: ArrayLike, add_intercept_column: bool = True) -> dict:
    """Return the projection matrix H and its idempotence check."""

    X = _as_2d_array(X)
    design = add_intercept(X) if add_intercept_column else X
    xtx = design.T @ design

    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("X'X is singular or ill-conditioned") from exc

    H = design @ xtx_inv @ design.T
    idempotent = np.allclose(H @ H, H)
    symmetric = np.allclose(H, H.T)
    return {
        "H": H,
        "is_idempotent": idempotent,
        "is_symmetric": symmetric,
        "rank": int(np.linalg.matrix_rank(H)),
    }


def model_metrics(y: ArrayLike, y_hat: ArrayLike, p: int) -> dict:

    y = _as_1d_array(y)
    y_hat = _as_1d_array(y_hat)
    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape")

    n_samples = y.shape[0]
    if p < 0:
        raise ValueError("p must be non-negative")

    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if tss == 0 else 1.0 - rss / tss

    denominator_df = n_samples - p - 1
    if denominator_df <= 0:
        adj_r2 = np.nan
        f_stat = np.nan
    else:
        adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1) / denominator_df
        if p == 0 or rss == 0:
            f_stat = np.inf if p > 0 else np.nan
        else:
            explained = (tss - rss) / p
            unexplained = rss / denominator_df
            f_stat = explained / unexplained if unexplained != 0 else np.inf

    return {
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "adjusted_r2": adj_r2,
        "f_statistic": f_stat,
    }


def _student_t_cdf(value: float, df: int) -> float:
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError("scipy is required for t-based inference") from exc
    return float(stats.t.cdf(value, df))


def _student_t_ppf(probability: float, df: int) -> float:
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError("scipy is required for confidence intervals") from exc
    return float(stats.t.ppf(probability, df))


def coef_inference(
    X: ArrayLike,
    y: ArrayLike,
    beta_hat: ArrayLike,
    sigma2_hat: float,
    add_intercept_column: bool = True,
    alpha: float = 0.05,
) -> dict:
    """Compute standard errors, t-statistics, p-values, and confidence intervals."""

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)
    design = add_intercept(X) if add_intercept_column else X

    if design.shape[1] != beta_hat.shape[0]:
        raise ValueError("beta_hat length must match the number of model parameters")

    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("X'X is singular or ill-conditioned") from exc

    covariance = sigma2_hat * xtx_inv
    standard_errors = np.sqrt(np.diag(covariance))
    t_stats = beta_hat / standard_errors
    df_resid = design.shape[0] - design.shape[1]
    if df_resid <= 0:
        raise ValueError("Not enough degrees of freedom for inference")

    p_values = np.array([
        2.0 * (1.0 - _student_t_cdf(abs(value), df_resid)) for value in t_stats
    ])
    t_crit = _student_t_ppf(1.0 - alpha / 2.0, df_resid)
    intervals = np.column_stack(
        (beta_hat - t_crit * standard_errors, beta_hat + t_crit * standard_errors)
    )

    return {
        "standard_errors": standard_errors,
        "t_statistics": t_stats,
        "p_values": p_values,
        "confidence_intervals": intervals,
        "df_resid": df_resid,
    }


def vif(X: ArrayLike, add_intercept_column: bool = True) -> dict:
    """Compute the variance inflation factor for each feature.

    The intercept, if added, is excluded from the returned values.
    """

    X = _as_2d_array(X)
    if add_intercept_column:
        X = X.copy()

    n_samples, n_features = X.shape
    if n_features < 2:
        raise ValueError("VIF requires at least two predictors")

    values = []
    for target_idx in range(n_features):
        target = X[:, target_idx]
        others = np.delete(X, target_idx, axis=1)
        others_design = add_intercept(others)
        auxiliary_result = ols_fit(others_design, target, add_intercept_column=False)
        target_hat = auxiliary_result["y_hat"]

        metrics = model_metrics(target, target_hat, p=others.shape[1])
        r2 = metrics["r2"]
        if np.isclose(1.0 - r2, 0.0):
            values.append(np.inf)
        else:
            values.append(1.0 / (1.0 - r2))

    return {
        "feature_indices": list(range(n_features)),
        "vif": np.asarray(values, dtype=float),
    }


def verify_ols_with_sklearn(X: ArrayLike, y: ArrayLike, add_intercept_column: bool = True) -> dict:
    """Verify the custom OLS result against sklearn.linear_model.LinearRegression."""

    from sklearn.linear_model import LinearRegression

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    custom_result = ols_fit(X, y, add_intercept_column=add_intercept_column)
    model = LinearRegression(fit_intercept=add_intercept_column)
    model.fit(X, y)

    if add_intercept_column:
        sklearn_beta = np.concatenate(
            ([model.intercept_], np.asarray(model.coef_, dtype=float).reshape(-1))
        )
    else:
        sklearn_beta = np.asarray(model.coef_, dtype=float).reshape(-1)

    sklearn_y_hat = model.predict(X)

    return {
        "custom_beta_hat": np.asarray(custom_result["beta_hat"], dtype=float).reshape(-1),
        "sklearn_beta_hat": sklearn_beta,
        "beta_close": bool(np.allclose(custom_result["beta_hat"], sklearn_beta)),
        "custom_y_hat": np.asarray(custom_result["y_hat"], dtype=float).reshape(-1),
        "sklearn_y_hat": np.asarray(sklearn_y_hat, dtype=float).reshape(-1),
        "y_hat_close": bool(np.allclose(custom_result["y_hat"], sklearn_y_hat)),
    }


__all__ = [
    "add_intercept",
    "coef_inference",
    "hat_matrix",
    "model_metrics",
    "ols_fit",
    "verify_ols_with_sklearn",
    "vif",
]

def _make_datasets():
    """Return two small synthetic datasets for demos/tests.

    - dataset A: single predictor with intercept y = 2*x + 1
    - dataset B: two predictors with known betas [1.0, 2.0] and intercept 0.5
    """
    X1 = np.arange(1, 6, dtype=float).reshape(-1, 1)
    y1 = 2.0 * X1[:, 0] + 1.0

    rng = np.random.default_rng(0)
    X2 = rng.normal(size=(8, 2))
    # set a ground-truth beta
    beta_true = np.array([0.5, 1.0, 2.0])  # intercept, b1, b2
    X2_design = add_intercept(X2)
    y2 = X2_design @ beta_true

    return (X1, y1), (X2, y2)

if __name__ == "__main__":
    # Inline demo and tests (no auxiliary functions) — two datasets
    (X1, y1), (X2, y2) = _make_datasets()
    results = {"passed": 0, "failed": 0, "details": []}

    def ok(name: str, condition: bool, note: str = ""):
        if condition:
            results["passed"] += 1
            results["details"].append((name, True, note))
            print(f"ĐẠT: {name} {note}")
        else:
            results["failed"] += 1
            results["details"].append((name, False, note))
            print(f"KHÔNG ĐẠT: {name} {note}")

    ai1 = add_intercept(X1)
    ai2 = add_intercept(X2)
    print("Kết quả add_intercept(X1):\n", ai1)
    ok("add_intercept_shape1", ai1.shape == (5, 2), "X1 -> (5,2)")
    print("Kết quả add_intercept(X2):\n", ai2)
    ok("add_intercept_shape2", ai2.shape == (8, 3), "X2 -> (8,3)")

    res1 = ols_fit(X1, y1)
    print("Hệ số ước lượng từ ols_fit(X1,y1):", res1["beta_hat"])
    ok("ols_fit_beta1", np.allclose(res1["beta_hat"], np.array([1.0, 2.0])), "exact fit for X1")
    res2 = ols_fit(X2, y2)
    print("Hệ số ước lượng từ ols_fit(X2,y2):", res2["beta_hat"])
    ok("ols_fit_beta2", np.allclose(res2["beta_hat"], np.array([0.5, 1.0, 2.0])), "recover beta_true")

    h1 = hat_matrix(X1)
    print("Thống kê ma trận chiếu (H): is_idempotent=", h1["is_idempotent"], ", rank=", h1["rank"])
    ok("hat_idempotent", h1["is_idempotent"], "H^2 == H")
    ok("hat_rank", h1["rank"] == res1["n_params"], "rank == p+1")

    # model_metrics: test perfect fit and noisy case
    mm1 = model_metrics(y1, res1["y_hat"], p=1)
    print("Kết quả model_metrics cho dataset 1:", mm1)
    ok("model_metrics_r2_perfect", np.isclose(mm1["r2"], 1.0), "perfect fit r2=1")

    # second test for model_metrics with noise
    rng = np.random.default_rng(42)
    y2_noisy = y2 + rng.normal(scale=0.1, size=y2.shape)
    res2_noisy = ols_fit(X2, y2_noisy)
    mm2 = model_metrics(y2_noisy, res2_noisy["y_hat"], p=2)
    print("Kết quả model_metrics cho dataset 2 (có nhiễu):", mm2)
    ok("model_metrics_r2_less1", mm2["r2"] < 1.0, "noisy fit r2 < 1")

    # coef_inference: use noisy fit to avoid zero variance
    ci1 = coef_inference(X2, y2_noisy, res2_noisy["beta_hat"], res2_noisy["sigma2_hat"], add_intercept_column=True)
    print("Kết quả phân tích hệ số (sai số chuẩn, t-stat) dataset 2:", ci1["standard_errors"], ci1["t_statistics"])
    ok("coef_inference_dfpos", ci1["df_resid"] > 0, "degrees of freedom > 0")

    # second coef_inference test on simple dataset with small noise
    X_small = np.arange(1, 10, dtype=float).reshape(-1, 1)
    y_small = 2.0 * X_small[:, 0] + 1.0 + rng.normal(scale=0.5, size=X_small.shape[0])
    res_small = ols_fit(X_small, y_small)
    ci2 = coef_inference(X_small, y_small, res_small["beta_hat"], res_small["sigma2_hat"], add_intercept_column=True)
    print("coef_inference nhỏ:", ci2["standard_errors"], ci2["t_statistics"])
    ok("coef_inference_small_dfpos", ci2["df_resid"] > 0, "df > 0")

    # hat_matrix: two tests
    h1 = hat_matrix(X1)
    print("Ma trận chiếu H cho X1: is_idempotent=", h1["is_idempotent"], ", rank=", h1["rank"])
    ok("hat_idempotent_X1", h1["is_idempotent"], "H^2 == H for X1")
    h2 = hat_matrix(X2)
    print("Ma trận chiếu H cho X2: is_idempotent=", h2["is_idempotent"], ", rank=", h2["rank"])
    ok("hat_idempotent_X2", h2["is_idempotent"], "H^2 == H for X2")

    # vif: two tests — random X2 and collinear case
    vf = vif(X2, add_intercept_column=False)
    print("VIF cho X2:", vf["vif"])
    ok("vif_length", len(vf["vif"]) == X2.shape[1], "one VIF per feature")

    # perfect collinearity -> infinite VIF
    X_col = np.column_stack((X2[:, 0], 2.0 * X2[:, 0]))
    vf_col = vif(X_col, add_intercept_column=False)
    print("VIF cho X_collinear:", vf_col["vif"])
    ok("vif_infinite", np.any(~np.isfinite(vf_col["vif"])) or np.any(np.isinf(vf_col["vif"])), "infinite VIF expected")

    vs = verify_ols_with_sklearn(X2, y2_noisy)
    print("So sánh với sklearn — beta (custom vs sklearn) on noisy data:", vs["custom_beta_hat"], vs["sklearn_beta_hat"])
    ok("verify_sklearn_beta_close", vs["beta_close"], "custom vs sklearn on noisy data")

    print("-- Tóm tắt --")
    print(f"Đã đạt: {results['passed']}, Không đạt: {results['failed']}")
    if results["failed"] > 0:
        raise SystemExit(1)