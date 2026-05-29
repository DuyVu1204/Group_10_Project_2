"""
model_comparison.py – Task 3: Models, Cross-Validation, Evaluation, Residual Analysis.

Tất cả thuật toán core (OLS, Ridge, Lasso, KFold) tự cài đặt bằng NumPy.
Chỉ dùng thư viện ngoài cho: pandas (bảng), matplotlib/scipy (vẽ / Q-Q).

API chính:
    fit_ols_full          – OLS trên tất cả features
    fit_ols_selected      – OLS trên tập selected_features
    fit_ridge_cv          – Ridge chọn lambda bằng KFold trên train
    fit_lasso_cv          – Lasso chọn lambda bằng KFold trên train
    evaluate_models       – Đánh giá trên test set (MAE, RMSE, R²)
    get_best_model        – Chọn model tốt nhất
    plot_cv_results       – Vẽ đường CV RMSE theo lambda
    plot_residuals_vs_fitted  – Residuals vs Fitted
    plot_qq_residuals         – Q-Q Plot
    plot_scale_location       – Scale-Location
    plot_cooks_distance       – Cook's Distance
    plot_feature_importance   – Top coefficients
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import mae, rmse, r2_score
from config import RANDOM_STATE, N_SPLITS


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    INTERNAL HELPER CLASSES                      ║
# ╚══════════════════════════════════════════════════════════════════╝

class LinearModel:
    """
    Wrapper giữ coefficients + intercept cho OLS / Ridge / Lasso.
    Tương thích giao diện: model.predict(X), model.coef_, model.intercept_.
    """

    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef_ = coef            # shape (p,)
        self.intercept_ = intercept  # scalar

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# ╔══════════════════════════════════════════════════════════════════╗
# ║              CORE ALGORITHMS – TỰ CÀI ĐẶT                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def _ols_solve(X: np.ndarray, y: np.ndarray) -> LinearModel:
    """
    Giải OLS bằng Normal Equation:  β = (X^T X)^{-1} X^T y
    Thêm cột 1 để tính intercept.
    """
    n, p = X.shape
    # Thêm cột bias (intercept)
    X_b = np.column_stack([np.ones(n), X])  # (n, p+1)
    # Normal equation
    XtX = X_b.T @ X_b        # (p+1, p+1)
    Xty = X_b.T @ y          # (p+1,)
    beta = np.linalg.solve(XtX, Xty)  # (p+1,)
    intercept = beta[0]
    coef = beta[1:]
    return LinearModel(coef, intercept)


def _ridge_solve(X: np.ndarray, y: np.ndarray, lam: float) -> LinearModel:
    """
    Ridge Regression:  β = (X^T X + λI)^{-1} X^T y
    Không regularize intercept.
    """
    n, p = X.shape
    X_b = np.column_stack([np.ones(n), X])
    XtX = X_b.T @ X_b
    # Penalty matrix: không phạt intercept (vị trí [0, 0])
    penalty = lam * np.eye(p + 1)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(XtX + penalty, X_b.T @ y)
    return LinearModel(beta[1:], beta[0])


def _soft_threshold(x: float, threshold: float) -> float:
    """Soft-thresholding operator cho Lasso."""
    if x > threshold:
        return x - threshold    
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0


def _lasso_solve(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> LinearModel:
    """
    Lasso Regression bằng Coordinate Descent (Bản tối ưu hóa tốc độ).
    """
    n, p = X.shape
    beta = np.zeros(p)
    intercept = np.mean(y)
    
    # Tiền tính toán norm bình phương cột
    X_norms = np.sum(X**2, axis=0) / n
    
    # Khởi tạo phần dư: y - intercept - X @ beta (beta ban đầu bằng 0)
    residual = y - intercept
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        # Cập nhật intercept
        mean_res = np.mean(residual)
        intercept += mean_res
        residual -= mean_res
        
        # Coordinate descent cho từng feature
        for j in range(p):
            old_beta_j = beta[j]
            z = X_norms[j]
            
            # Tính rho = (X[:, j]^T @ residual) / n + X_norms[j] * old_beta_j
            rho = (X[:, j] @ residual) / n + z * old_beta_j
            
            if z == 0:
                new_beta_j = 0.0
            else:
                new_beta_j = _soft_threshold(rho, lam) / z
                
            if new_beta_j != old_beta_j:
                beta[j] = new_beta_j
                # Cập nhật phần dư tăng dần để tránh nhân ma trận X @ beta
                residual -= (new_beta_j - old_beta_j) * X[:, j]
                
        # Kiểm tra hội tụ
        if np.max(np.abs(beta - beta_old)) < tol:
            break
            
    return LinearModel(beta, intercept)


def _kfold_indices(n: int, n_splits: int, shuffle: bool = True,
                   random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Tạo KFold indices tự cài đặt.
    Return: list[(train_idx, val_idx)]
    """
    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current: current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_idx, val_idx))
        current += fold_size

    return folds


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      PUBLIC API – MODELS                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def fit_ols_full(X_train, y_train) -> LinearModel:
    """
    Fit OLS trên tất cả features.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray

    Returns
    -------
    LinearModel
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    return _ols_solve(X, y)


def fit_ols_selected(X_train, y_train, selected_features: list[str]) -> LinearModel:
    """
    Fit OLS chỉ trên các selected_features (VIF / p-value selection).

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series or np.ndarray
    selected_features : list[str]

    Returns
    -------
    LinearModel
    """
    X = np.asarray(X_train[selected_features], dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    return _ols_solve(X, y)


def fit_ridge_cv(
    X_train,
    y_train,
    lambdas: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[LinearModel, float, pd.DataFrame]:
    """
    Ridge Regression với KFold Cross-Validation chọn lambda.

    Chỉ dùng training set cho cross-validation. Không dùng test set.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    lambdas : array-like – lambda grid
    n_splits : int
    random_state : int

    Returns
    -------
    final_model : LinearModel – Ridge fit trên toàn bộ train
    best_lambda : float
    cv_results_df : pd.DataFrame – columns [lambda, mean_cv_rmse, std_cv_rmse]
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    n = X.shape[0]

    folds = _kfold_indices(n, n_splits, shuffle=True, random_state=random_state)

    records = []
    for lam in lambdas:
        fold_rmses = []
        for train_idx, val_idx in folds:
            model = _ridge_solve(X[train_idx], y[train_idx], lam)
            y_val_pred = model.predict(X[val_idx])
            fold_rmses.append(rmse(y[val_idx], y_val_pred))
        records.append({
            "lambda": lam,
            "mean_cv_rmse": np.mean(fold_rmses),
            "std_cv_rmse": np.std(fold_rmses),
        })

    cv_results_df = pd.DataFrame(records)
    best_idx = cv_results_df["mean_cv_rmse"].idxmin()
    best_lambda = cv_results_df.loc[best_idx, "lambda"]

    # Fit final model trên toàn bộ training set
    final_model = _ridge_solve(X, y, best_lambda)

    return final_model, best_lambda, cv_results_df


def fit_lasso_cv(
    X_train,
    y_train,
    lambdas: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[LinearModel, float, pd.DataFrame]:
    """
    Lasso Regression với KFold Cross-Validation chọn lambda.

    Chỉ dùng training set cho cross-validation. Không dùng test set.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    lambdas : array-like – lambda grid
    n_splits : int
    random_state : int

    Returns
    -------
    final_model : LinearModel – Lasso fit trên toàn bộ train
    best_lambda : float
    cv_results_df : pd.DataFrame – columns [lambda, mean_cv_rmse, std_cv_rmse]
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    n = X.shape[0]

    folds = _kfold_indices(n, n_splits, shuffle=True, random_state=random_state)

    records = []
    for lam in lambdas:
        fold_rmses = []
        for train_idx, val_idx in folds:
            model = _lasso_solve(X[train_idx], y[train_idx], lam)
            y_val_pred = model.predict(X[val_idx])
            fold_rmses.append(rmse(y[val_idx], y_val_pred))
        records.append({
            "lambda": lam,
            "mean_cv_rmse": np.mean(fold_rmses),
            "std_cv_rmse": np.std(fold_rmses),
        })

    cv_results_df = pd.DataFrame(records)
    best_idx = cv_results_df["mean_cv_rmse"].idxmin()
    best_lambda = cv_results_df.loc[best_idx, "lambda"]

    # Fit final model trên toàn bộ training set
    final_model = _lasso_solve(X, y, best_lambda)

    return final_model, best_lambda, cv_results_df


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   EVALUATION & COMPARISON                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def evaluate_models(
    models: dict[str, LinearModel],
    X_test,
    y_test,
) -> pd.DataFrame:
    """
    Đánh giá nhiều models trên test set.

    Parameters
    ----------
    models : dict[str, LinearModel]
        VD: {"OLS Full": model1, "OLS Selected": model2, ...}
        Nếu model được fit trên selected features thì X_test phải đã
        được lọc cùng cột TRƯỚC khi truyền vào, HOẶC truyền riêng.
        → Để linh hoạt, truyền dict {"name": (model, X_test_for_that_model)}.

        **Cách dùng đơn giản** (cùng X_test cho tất cả):
            models = {"OLS Full": model1, "Ridge": model2}
            evaluate_models(models, X_test, y_test)

    X_test : pd.DataFrame or np.ndarray or dict
        Nếu là dict thì key trùng với models, mỗi value là X_test tương ứng.
    y_test : pd.Series or np.ndarray

    Returns
    -------
    pd.DataFrame – columns [Model, MAE, RMSE, R2]
    """
    y_true = np.asarray(y_test, dtype=float).ravel()
    rows = []

    for name, model in models.items():
        if isinstance(X_test, dict):
            X = np.asarray(X_test[name], dtype=float)
        else:
            X = np.asarray(X_test, dtype=float)

        y_pred = model.predict(X)
        rows.append({
            "Model": name,
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        })

    return pd.DataFrame(rows)


def get_best_model(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
) -> str:
    """
    Trả về tên model tốt nhất theo metric.

    Parameters
    ----------
    results_df : pd.DataFrame – output từ evaluate_models
    metric : str – "RMSE", "MAE" (nhỏ hơn tốt hơn) hoặc "R2" (lớn hơn tốt hơn)

    Returns
    -------
    str – Tên model tốt nhất
    """
    if metric == "R2":
        idx = results_df[metric].idxmax()
    else:
        idx = results_df[metric].idxmin()
    return results_df.loc[idx, "Model"]


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   PLOTTING – CV RESULTS                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def plot_cv_results(
    cv_results_df: pd.DataFrame,
    title: str = "Cross-Validation RMSE vs Lambda",
):
    """
    Vẽ đường CV RMSE ± 1 std theo lambda (log scale).

    Parameters
    ----------
    cv_results_df : pd.DataFrame – columns [lambda, mean_cv_rmse, std_cv_rmse]
    title : str
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    lam = cv_results_df["lambda"]
    mean_rmse = cv_results_df["mean_cv_rmse"]
    std_rmse = cv_results_df["std_cv_rmse"]

    ax.plot(lam, mean_rmse, "o-", color="#2563eb", linewidth=1.5, markersize=3,
            label="Mean CV RMSE")
    ax.fill_between(lam, mean_rmse - std_rmse, mean_rmse + std_rmse,
                    alpha=0.2, color="#2563eb", label="± 1 std")

    # Đánh dấu best lambda
    best_idx = mean_rmse.idxmin()
    best_lam = lam[best_idx]
    best_rmse = mean_rmse[best_idx]
    ax.axvline(best_lam, color="#dc2626", linestyle="--", linewidth=1,
               label=f"Best λ = {best_lam:.4g}")
    ax.plot(best_lam, best_rmse, "s", color="#dc2626", markersize=8, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Lambda (λ)", fontsize=12)
    ax.set_ylabel("CV RMSE", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                 RESIDUAL DIAGNOSTIC PLOTS                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def plot_residuals_vs_fitted(y_true, y_pred, title: str = "Residuals vs Fitted"):
    """
    Scatter plot: ŷ (x-axis) vs residual (y-axis).
    Kiểm tra: tính tuyến tính, homoscedasticity.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, edgecolors="k", linewidths=0.3,
               color="#6366f1", s=20)
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1)

    # LOWESS smoothing line (tự cài đặt đơn giản)
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    res_sorted = residuals[sorted_idx]
    window = max(len(res_sorted) // 20, 10)
    smooth = np.convolve(res_sorted, np.ones(window) / window, mode="same")
    ax.plot(y_pred_sorted, smooth, color="#f59e0b", linewidth=2, label="Smoothed")

    ax.set_xlabel("Fitted Values (ŷ)", fontsize=12)
    ax.set_ylabel("Residuals (y − ŷ)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_qq_residuals(y_true, y_pred, title: str = "Q-Q Plot of Residuals"):
    """
    Normal Q-Q plot.
    Kiểm tra: tính chuẩn của residuals.
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    # Standardize residuals
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(std_res, dist="norm", plot=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.get_lines()[0].set(color="#6366f1", markersize=4, alpha=0.6)
    ax.get_lines()[1].set(color="#dc2626", linewidth=1.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_scale_location(y_true, y_pred, title: str = "Scale-Location Plot"):
    """
    sqrt(|standardized residuals|) vs fitted values.
    Kiểm tra: homoscedasticity.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    sqrt_abs_std_res = np.sqrt(np.abs(std_res))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, sqrt_abs_std_res, alpha=0.4, edgecolors="k",
               linewidths=0.3, color="#10b981", s=20)

    # Smoothing line
    sorted_idx = np.argsort(y_pred)
    yp = y_pred[sorted_idx]
    sr = sqrt_abs_std_res[sorted_idx]
    window = max(len(sr) // 20, 10)
    smooth = np.convolve(sr, np.ones(window) / window, mode="same")
    ax.plot(yp, smooth, color="#f59e0b", linewidth=2, label="Smoothed")

    ax.set_xlabel("Fitted Values (ŷ)", fontsize=12)
    ax.set_ylabel("√|Standardized Residuals|", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_cooks_distance(X, y_true, y_pred, title: str = "Cook's Distance"):
    """
    Cook's Distance cho mỗi observation.

    D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)

    với h_ii = diagonal của Hat matrix H = X(X^T X)^{-1} X^T

    Parameters
    ----------
    X : array-like – design matrix (đã processed, KHÔNG cần cột 1)
    y_true : array-like
    y_pred : array-like
    """
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    n, p_features = X.shape
    residuals = y_true - y_pred

    # Thêm cột intercept
    X_b = np.column_stack([np.ones(n), X])
    p = X_b.shape[1]  # p + 1

    # Hat matrix diagonal: h_ii
    try:
        XtX_inv = np.linalg.inv(X_b.T @ X_b)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_b.T @ X_b)
    H = X_b @ XtX_inv @ X_b.T
    h = np.diag(H)

    # MSE
    mse = np.sum(residuals ** 2) / (n - p)

    # Cook's distance
    cooks_d = (residuals ** 2) / (p * mse + 1e-12) * (h / (1 - h + 1e-12) ** 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stem(np.arange(n), cooks_d, linefmt="#6366f1", markerfmt="o",
            basefmt=" ", label="Cook's Distance")
    ax.axhline(4 / n, color="#dc2626", linestyle="--", linewidth=1,
               label=f"Threshold = 4/n = {4/n:.4f}")

    # Highlight influential points
    influential = np.where(cooks_d > 4 / n)[0]
    if len(influential) > 0 and len(influential) <= 20:
        for idx in influential:
            ax.annotate(str(idx), (idx, cooks_d[idx]),
                        fontsize=7, color="#dc2626", ha="center", va="bottom")

    ax.set_xlabel("Observation Index", fontsize=12)
    ax.set_ylabel("Cook's Distance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   FEATURE IMPORTANCE PLOT                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def plot_feature_importance(
    model: LinearModel,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance (Coefficients)",
):
    """
    Vẽ barplot top coefficients (absolute value) với màu theo dấu.

    Parameters
    ----------
    model : LinearModel
    feature_names : list[str]
    top_n : int – Hiện top_n features
    title : str
    """
    coef = model.coef_
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coef": np.abs(coef),
    })
    df = df.sort_values("abs_coef", ascending=False).head(top_n)
    df = df.sort_values("abs_coef", ascending=True)  # cho barh đẹp

    colors = ["#dc2626" if c < 0 else "#2563eb" for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(df["feature"], df["coefficient"], color=colors, edgecolor="white",
            linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", label="Positive"),
        Patch(facecolor="#dc2626", label="Negative"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.tight_layout()
    plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║              CONVENIENCE: RUN ALL DIAGNOSTICS                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_residual_diagnostics(model: LinearModel, X, y_true,
                             model_name: str = "Best Model"):
    """
    Chạy toàn bộ 4 residual diagnostic plots cho một model.
    """
    X_arr = np.asarray(X, dtype=float)
    y_pred = model.predict(X_arr)

    print(f"\n{'='*60}")
    print(f"  Residual Diagnostics -- {model_name}")
    print(f"{'='*60}\n")

    plot_residuals_vs_fitted(y_true, y_pred,
                             title=f"Residuals vs Fitted — {model_name}")
    plot_qq_residuals(y_true, y_pred,
                      title=f"Q-Q Plot — {model_name}")
    plot_scale_location(y_true, y_pred,
                        title=f"Scale-Location — {model_name}")
    plot_cooks_distance(X_arr, y_true, y_pred,
                        title=f"Cook's Distance — {model_name}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   EXAMPLE / STANDALONE TEST                     ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Quick sanity check với dữ liệu tổng hợp nhỏ
    np.random.seed(RANDOM_STATE)
    n, p = 200, 5
    X_synth = np.random.randn(n, p)
    true_coef = np.array([3.0, -1.5, 0.0, 2.0, -0.5])
    y_synth = X_synth @ true_coef + 7.0 + np.random.randn(n) * 2

    # Split 80/20
    split = int(0.8 * n)
    X_tr, X_te = X_synth[:split], X_synth[split:]
    y_tr, y_te = y_synth[:split], y_synth[split:]

    print("-- OLS Full --")
    m_ols = fit_ols_full(X_tr, y_tr)
    print(f"  coef = {m_ols.coef_}")
    print(f"  intercept = {m_ols.intercept_:.4f}")

    import numpy as _np
    lambdas = _np.logspace(-4, 4, 30)

    print("\n-- Ridge CV --")
    m_ridge, best_lam_r, cv_r = fit_ridge_cv(X_tr, y_tr, lambdas)
    print(f"  best_lambda = {best_lam_r:.6f}")

    print("\n-- Lasso CV --")
    m_lasso, best_lam_l, cv_l = fit_lasso_cv(X_tr, y_tr, lambdas)
    print(f"  best_lambda = {best_lam_l:.6f}")

    models = {
        "OLS Full": m_ols,
        "Ridge CV": m_ridge,
        "Lasso CV": m_lasso,
    }
    results = evaluate_models(models, X_te, y_te)
    print("\n-- Evaluation on Test Set --")
    print(results.to_string(index=False))
    print(f"\nBest model: {get_best_model(results, 'RMSE')}")
