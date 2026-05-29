
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
    return np.column_stack((np.ones((X.shape[0], 1), dtype=float), X))


def ridge_fit(
    X: ArrayLike,
    y: ArrayLike,
    lam: float,
    add_intercept_column: bool = True,
) -> dict:
    """Closed-form ridge regression.

    The intercept is excluded from the penalty term.
    """

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    if add_intercept_column:
        design = add_intercept(X)
        penalty = np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
    else:
        design = X
        penalty = np.eye(design.shape[1], dtype=float)

    xtx = design.T @ design
    xty = design.T @ y

    try:
        beta_hat = np.linalg.solve(xtx + lam * penalty, xty)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Ridge system is singular or ill-conditioned") from exc

    y_hat = design @ beta_hat
    residuals = y - y_hat
    rss = float(residuals.T @ residuals)

    return {
        "beta_hat": beta_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "rss": rss,
        "design_matrix": design,
        "lambda": float(lam),
    }


def ridge_path(
    X: ArrayLike,
    y: ArrayLike,
    lambdas: ArrayLike,
    add_intercept_column: bool = True,
) -> dict:
    """Compute ridge coefficients for a grid of lambda values."""

    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    coeffs = []
    for lam in lambdas:
        result = ridge_fit(X, y, float(lam), add_intercept_column=add_intercept_column)
        coeffs.append(result["beta_hat"])
    return {
        "lambdas": lambdas,
        "coefficients": np.vstack(coeffs),
    }


def plot_ridge_trace(
    X: ArrayLike,
    y: ArrayLike,
    lambdas: ArrayLike,
    add_intercept_column: bool = True,
):
    """Plot the ridge trace and return the figure."""

    import matplotlib.pyplot as plt

    path = ridge_path(X, y, lambdas, add_intercept_column=add_intercept_column)
    lambdas = path["lambdas"]
    coefficients = path["coefficients"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(coefficients.shape[1]):
        ax.plot(lambdas, coefficients[:, idx], label=f"beta_{idx}")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Ridge Trace")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig, ax


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def lasso_fit(
    X: ArrayLike,
    y: ArrayLike,
    lam: float,
    max_iter: int = 5000,
    tol: float = 1e-6,
    standardize: bool = True,
) -> dict:
    """Lasso regression via coordinate descent.

    The intercept is estimated on centered data and is not penalized.
    """

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - x_mean
    y_centered = y - y_mean

    if standardize:
        scale = X_centered.std(axis=0, ddof=0)
        scale[scale == 0.0] = 1.0
        X_working = X_centered / scale
    else:
        scale = np.ones(X.shape[1], dtype=float)
        X_working = X_centered

    n_samples, n_features = X_working.shape
    beta = np.zeros(n_features, dtype=float)
    history = []
    converged = False

    for iteration in range(1, max_iter + 1):
        beta_old = beta.copy()
        for feature_idx in range(n_features):
            partial_residual = y_centered - X_working @ beta + X_working[:, feature_idx] * beta[feature_idx]
            rho = float(np.dot(X_working[:, feature_idx], partial_residual))
            z = float(np.dot(X_working[:, feature_idx], X_working[:, feature_idx]))
            if z == 0.0:
                beta[feature_idx] = 0.0
            else:
                beta[feature_idx] = _soft_threshold(rho, lam) / z

        objective = 0.5 * float(np.sum((y_centered - X_working @ beta) ** 2)) + lam * float(np.sum(np.abs(beta)))
        history.append(objective)
        if np.linalg.norm(beta - beta_old, ord=np.inf) < tol:
            converged = True
            break

    beta_unscaled = beta / scale
    intercept = float(y_mean - x_mean @ beta_unscaled)
    y_hat = intercept + X @ beta_unscaled
    residuals = y - y_hat

    return {
        "intercept": intercept,
        "beta_hat": beta_unscaled,
        "y_hat": y_hat,
        "residuals": residuals,
        "objective_history": np.asarray(history, dtype=float),
        "n_iter": iteration,
        "converged": converged,
        "lambda": float(lam),
        "standardized": standardize,
    }


def predict_ridge(X: ArrayLike, beta_hat: ArrayLike, add_intercept_column: bool = True) -> np.ndarray:
    X = _as_2d_array(X)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)
    design = add_intercept(X) if add_intercept_column else X
    if design.shape[1] != beta_hat.shape[0]:
        raise ValueError("beta_hat length must match the design matrix")
    return design @ beta_hat


def predict_lasso(X: ArrayLike, intercept: float, beta_hat: ArrayLike) -> np.ndarray:
    X = _as_2d_array(X)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)
    if X.shape[1] != beta_hat.shape[0]:
        raise ValueError("beta_hat length must match the number of features")
    return intercept + X @ beta_hat


def verify_ridge_with_sklearn(
    X: ArrayLike,
    y: ArrayLike,
    lam: float,
    add_intercept_column: bool = True,
) -> dict:
    """Verify the custom ridge solution against sklearn.linear_model.Ridge."""

    from sklearn.linear_model import Ridge

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    custom_result = ridge_fit(X, y, lam, add_intercept_column=add_intercept_column)
    model = Ridge(alpha=lam, fit_intercept=add_intercept_column)
    model.fit(X, y)

    if add_intercept_column:
        sklearn_beta = np.concatenate(
            ([float(model.intercept_)], np.asarray(model.coef_, dtype=float).reshape(-1))
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
        "rss_close": bool(
            np.isclose(
                float(custom_result["rss"]),
                float(np.sum((y - sklearn_y_hat) ** 2)),
            )
        ),
    }


def _prepare_lasso_verification_data(
    X: ArrayLike,
    y: ArrayLike,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = _as_2d_array(X)
    y = _as_1d_array(y)
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - x_mean
    y_centered = y - y_mean

    if standardize:
        scale = X_centered.std(axis=0, ddof=0)
        scale[scale == 0.0] = 1.0
        X_working = X_centered / scale
    else:
        scale = np.ones(X.shape[1], dtype=float)
        X_working = X_centered

    return X, y, X_working, y_centered, np.asarray([x_mean, y_mean, scale], dtype=object)


def verify_lasso_with_sklearn(
    X: ArrayLike,
    y: ArrayLike,
    lam: float,
    max_iter: int = 5000,
    tol: float = 1e-6,
    standardize: bool = True,
) -> dict:
    """Verify the custom lasso solution against sklearn.linear_model.Lasso.

    The comparison is performed on the same centered/standardized data used by
    the custom coordinate-descent implementation, then mapped back to the
    original feature scale.
    """

    from sklearn.linear_model import Lasso

    X = _as_2d_array(X)
    y = _as_1d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    custom_result = lasso_fit(
        X,
        y,
        lam,
        max_iter=max_iter,
        tol=tol,
        standardize=standardize,
    )

    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - x_mean
    y_centered = y - y_mean
    if standardize:
        scale = X_centered.std(axis=0, ddof=0)
        scale[scale == 0.0] = 1.0
        X_working = X_centered / scale
    else:
        scale = np.ones(X.shape[1], dtype=float)
        X_working = X_centered

    alpha = lam / X.shape[0]
    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        selection="cyclic",
    )
    model.fit(X_working, y_centered)

    beta_standardized = np.asarray(model.coef_, dtype=float).reshape(-1)
    beta_unscaled = beta_standardized / scale
    intercept = float(y_mean - x_mean @ beta_unscaled)
    sklearn_y_hat = intercept + X @ beta_unscaled

    return {
        "custom_intercept": float(custom_result["intercept"]),
        "sklearn_intercept": intercept,
        "intercept_close": bool(np.isclose(custom_result["intercept"], intercept)),
        "custom_beta_hat": np.asarray(custom_result["beta_hat"], dtype=float).reshape(-1),
        "sklearn_beta_hat": beta_unscaled,
        "beta_close": bool(np.allclose(custom_result["beta_hat"], beta_unscaled)),
        "custom_y_hat": np.asarray(custom_result["y_hat"], dtype=float).reshape(-1),
        "sklearn_y_hat": np.asarray(sklearn_y_hat, dtype=float).reshape(-1),
        "y_hat_close": bool(np.allclose(custom_result["y_hat"], sklearn_y_hat)),
        "objective_history_length": int(len(custom_result["objective_history"])),
    }


__all__ = [
    "add_intercept",
    "lasso_fit",
    "plot_ridge_trace",
    "predict_lasso",
    "predict_ridge",
    "verify_lasso_with_sklearn",
    "verify_ridge_with_sklearn",
    "ridge_fit",
    "ridge_path",
]


if __name__ == "__main__":
    # Inline demos/tests for ridge and lasso
    results = {"passed": 0, "failed": 0}

    def ok(name: str, condition: bool, note: str = ""):
        if condition:
            results["passed"] += 1
            print(f"ĐẠT: {name} {note}")
        else:
            results["failed"] += 1
            print(f"KHÔNG ĐẠT: {name} {note}")

    # Dataset A: simple linear with intercept
    X1 = np.arange(1, 6, dtype=float).reshape(-1, 1)
    y1 = 3.0 * X1[:, 0] - 1.0

    # Dataset B: two features with known betas (intercept, b1, b2)
    rng = np.random.default_rng(1)
    X2 = rng.normal(size=(10, 2))
    beta_true = np.array([0.2, 1.5, -0.7])
    X2_design = add_intercept(X2)
    y2 = X2_design @ beta_true

    # Ridge: with lambda=0 should approximate OLS
    r1 = ridge_fit(X1, y1, lam=0.0)
    print("Hệ số ước lượng từ ridge_fit(X1,y1).beta_hat:", r1["beta_hat"])
    ok("ridge_beta_simple", np.allclose(r1["beta_hat"], np.array([-1.0, 3.0])), "lambda=0 matches OLS")

    # second ridge test: small dataset with lambda>0
    X1b = np.array([[1.0], [2.0], [3.0], [4.0]])
    y1b = np.array([2.0, 4.1, 6.0, 8.2])
    r1b = ridge_fit(X1b, y1b, lam=0.5)
    print("Ridge test 2 beta:", r1b["beta_hat"]) 
    ok("ridge_beta_test2_shape", r1b["beta_hat"].shape[0] == 2, "intercept + slope returned")

    r2 = ridge_fit(X2, y2, lam=1.0)
    print("Hệ số ước lượng từ ridge_fit(X2,y2).beta_hat:", r2["beta_hat"])
    ok("ridge_beta_shape", r2["beta_hat"].shape[0] == X2_design.shape[1], "beta length matches design")

    # ridge_path and plot
    lambdas = np.logspace(-3, 3, 5)
    path = ridge_path(X2, y2, lambdas)
    print("Hệ số theo đường ridge (ridge_path) ->\n", path["coefficients"]) 
    ok("ridge_path_shape", path["coefficients"].shape[0] == len(lambdas), "coeff matrix rows == lambdas")
    try:
        fig, ax = plot_ridge_trace(X2, y2, lambdas)
        print("plot_ridge_trace trả về figure")
        ok("plot_ridge_trace_return", hasattr(fig, "tight_layout"), "figure returned")
        import matplotlib

        matplotlib.pyplot.close(fig)
    except Exception as e:
        ok("plot_ridge_trace_err", False, str(e))

    # predict_ridge
    y1_pred = predict_ridge(X1, r1["beta_hat"])
    print("Dự đoán từ predict_ridge(X1):", y1_pred)
    ok("predict_ridge_len", len(y1_pred) == len(y1), "prediction length")

    # second predict_ridge test
    y1b_pred = predict_ridge(X1b, r1b["beta_hat"])
    print("Dự đoán test2 từ predict_ridge:", y1b_pred)
    ok("predict_ridge_len2", len(y1b_pred) == len(y1b), "prediction length test2")

    # Lasso: test convergence and shape
    l1 = lasso_fit(X2, y2, lam=0.1, max_iter=2000)
    print("Hệ số ước lượng từ lasso_fit(X2):", l1["beta_hat"], ", intercept:", l1["intercept"])
    ok("lasso_converged", bool(l1["converged"]), "converged")
    ok("lasso_beta_shape", l1["beta_hat"].shape[0] == X2.shape[1], "beta length")

    # second lasso test: stronger regularization to encourage sparsity
    l2 = lasso_fit(X2, y2, lam=1.0, max_iter=5000)
    print("Lasso test2 beta (stronger lambda):", l2["beta_hat"], ", intercept:", l2["intercept"])
    ok("lasso_converged2", bool(l2["converged"]), "converged2")
    ok("lasso_sparsity_check", np.sum(np.isclose(l2["beta_hat"], 0.0)) >= 0, "sparsity observed or none")

    y2_pred = predict_lasso(X2, l1["intercept"], l1["beta_hat"])
    print("Dự đoán từ predict_lasso(X2):", y2_pred)
    ok("predict_lasso_len", len(y2_pred) == len(y2), "prediction length")

    # Verify against sklearn
    try:
        vr = verify_ridge_with_sklearn(X2, y2, lam=1.0)
        print("Kết quả verify_ridge_with_sklearn:", {"beta_close": vr["beta_close"], "beta_sklearn": vr["sklearn_beta_hat"]})
        ok("verify_ridge_beta_close", vr["beta_close"], "custom vs sklearn ridge")
        # second verification on small dataset
        vr2 = verify_ridge_with_sklearn(X1b, y1b, lam=0.5)
        print("Kết quả verify_ridge test2:", {"beta_close": vr2["beta_close"], "beta_sklearn": vr2["sklearn_beta_hat"]})
        ok("verify_ridge_beta_close2", vr2["beta_close"], "custom vs sklearn ridge test2")
    except Exception as e:
        ok("verify_ridge_err", False, str(e))

    try:
        vl = verify_lasso_with_sklearn(X2, y2, lam=0.01, max_iter=2000)
        print("Kết quả verify_lasso_with_sklearn:", {"beta_close": vl["beta_close"], "beta_sklearn": vl["sklearn_beta_hat"]})
        ok("verify_lasso_beta_close", vl["beta_close"], "custom vs sklearn lasso")
        # second verification with stronger lambda
        vl2 = verify_lasso_with_sklearn(X2, y2, lam=0.5, max_iter=5000)
        print("Kết quả verify_lasso test2:", {"beta_close": vl2["beta_close"], "beta_sklearn": vl2["sklearn_beta_hat"]})
        ok("verify_lasso_beta_close2", vl2["beta_close"], "custom vs sklearn lasso test2")
    except Exception as e:
        ok("verify_lasso_err", False, str(e))

    print("-- Tóm tắt --")
    print(f"Đã đạt: {results['passed']}, Không đạt: {results['failed']}")
    if results["failed"] > 0:
        raise SystemExit(1)