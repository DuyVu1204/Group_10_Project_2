import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None


def residual_plots(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray):
    """
    Vẽ 4 biểu đồ phân tích phần dư OLS:
    1. Residuals vs Fitted
    2. Normal Q-Q Plot
    3. Scale-Location
    4. Cook's Distance

    Core implementation:
        - Tự tính fitted values
        - Tự tính residuals
        - Tự tính hat matrix / leverage
        - Tự tính standardized residuals
        - Tự tính Cook's distance

    sklearn (nếu có) chỉ dùng ở phần verification.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X phải là numpy.ndarray.")
    if not isinstance(y, np.ndarray):
        raise ValueError("y phải là numpy.ndarray.")
    if not isinstance(beta_hat, np.ndarray):
        raise ValueError("beta_hat phải là numpy.ndarray.")
    if X.ndim != 2:
        raise ValueError("X phải là ma trận 2 chiều (ndim == 2).")
    if y.ndim != 1:
        raise ValueError("y phải là vector 1 chiều (ndim == 1).")
    if beta_hat.ndim != 1:
        raise ValueError("beta_hat phải là vector 1 chiều (ndim == 1).")

    n = X.shape[0]
    p = beta_hat.shape[0]

    if n == 0 or X.shape[1] == 0:
        raise ValueError("X không được rỗng.")
    if y.shape[0] != n:
        raise ValueError("Số mẫu của X và y phải khớp.")
    if not (X.shape[1] + 1 == p or X.shape[1] == p):
        raise ValueError(
            "Kích thước beta_hat không phù hợp với X "
            "(phải là số biến + 1 hoặc đúng số biến nếu không có intercept)."
        )
    if n <= p:
        raise ValueError("Cần số mẫu n > số tham số p để phân tích residual ổn định.")

    if X.shape[1] + 1 == p:
        X_ = np.hstack([np.ones((n, 1)), X])
    else:
        X_ = X

    y_hat = X_ @ beta_hat
    residuals = y - y_hat

    XtX = X_.T @ X_
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError("Ma trận X^T X suy biến, không thể tính hat matrix duy nhất.")

    H = X_ @ np.linalg.solve(XtX, X_.T)
    h_ii = np.diag(H)

    rss = np.sum(residuals ** 2)
    sigma2_hat = rss / (n - p)

    std_residuals = residuals / (np.sqrt(sigma2_hat * (1.0 - h_ii)) + 1e-12)

    cooks_d = (
        (residuals ** 2) / (p * sigma2_hat + 1e-12)
        * (h_ii / ((1.0 - h_ii) ** 2 + 1e-12))
    )

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(y_hat, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")

    plt.subplot(2, 2, 2)
    sorted_std_residuals = np.sort(std_residuals)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = stats.norm.ppf(probabilities)
    plt.scatter(theoretical_quantiles, sorted_std_residuals, alpha=0.7)
    max_val = max(np.max(theoretical_quantiles), np.max(sorted_std_residuals))
    min_val = min(np.min(theoretical_quantiles), np.min(sorted_std_residuals))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.title("Normal Q-Q")

    plt.subplot(2, 2, 3)
    sqrt_abs_std_residuals = np.sqrt(np.abs(std_residuals))
    plt.scatter(y_hat, sqrt_abs_std_residuals, alpha=0.7)
    plt.xlabel("Fitted values")
    plt.ylabel(r"$\sqrt{|Standardized\ Residuals|}$")
    plt.title("Scale-Location")

    plt.subplot(2, 2, 4)
    plt.stem(np.arange(n), cooks_d, markerfmt=",", basefmt=" ")
    plt.axhline(4 / n, color="red", linestyle="--", label="Threshold 4/n")
    plt.xlabel("Observation number")
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        "residuals": residuals,
        "standardized_residuals": std_residuals,
        "hat_values": h_ii,
        "cooks_distance": cooks_d,
        "sigma2_hat": sigma2_hat,
        "y_hat": y_hat,
    }


def _verification_with_sklearn(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Verification helper dùng sklearn để đối chiếu fitted values và residuals.
    Chỉ dùng cho kiểm chứng, không dùng trong core implementation.
    """
    if LinearRegression is None:
        raise ImportError("Cần cài scikit-learn để chạy verification.")

    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat

    return {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "y_hat": y_hat,
        "residuals": residuals,
    }


def test_residual_plots_no_error():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(12, 2))
    beta_true = np.array([1.0, 2.0, -1.0])
    X_ = np.hstack([np.ones((12, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.1, size=12)
    beta_hat = np.linalg.solve(X_.T @ X_, X_.T @ y)
    try:
        plt.close("all")
        residual_plots(X, y, beta_hat)
        plt.close("all")
    except Exception as e:
        assert False, f"residual_plots bị lỗi: {e}"


def test_residual_mean_near_zero():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(30, 2))
    beta_true = np.array([1.5, -2.5, 0.5])
    X_ = np.hstack([np.ones((30, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.05, size=30)
    beta_hat = np.linalg.solve(X_.T @ X_, X_.T @ y)
    results = residual_plots(X, y, beta_hat)
    plt.close("all")
    assert np.isclose(np.mean(results["residuals"]), 0.0, atol=1e-10)


def test_residual_plots_with_outlier():
    rng = np.random.default_rng(789)
    X = rng.normal(size=(20, 2))
    beta_true = np.array([1.0, 2.0, -1.0])
    X_ = np.hstack([np.ones((20, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.1, size=20)
    y[0] += 10.0
    beta_hat = np.linalg.solve(X_.T @ X_, X_.T @ y)
    plt.close("all")
    results = residual_plots(X, y, beta_hat)
    plt.close("all")
    assert np.max(results["cooks_distance"]) > 1


def test_leverage_property():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(25, 4))
    beta_true = np.array([1.0, -2.0, 0.5, 1.5, -0.7])
    X_ = np.hstack([np.ones((25, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.2, size=25)
    beta_hat = np.linalg.solve(X_.T @ X_, X_.T @ y)
    results = residual_plots(X, y, beta_hat)
    plt.close("all")
    h_ii = results["hat_values"]
    assert np.all(h_ii >= -1e-10)
    assert np.all(h_ii <= 1 + 1e-10)


def test_leverage_trace_property():
    rng = np.random.default_rng(2027)
    X = rng.normal(size=(40, 3))
    beta_true = np.array([1.0, 2.0, -1.0, 0.5])
    X_ = np.hstack([np.ones((40, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.1, size=40)
    beta_hat = np.linalg.solve(X_.T @ X_, X_.T @ y)
    results = residual_plots(X, y, beta_hat)
    plt.close("all")
    leverage_sum = np.sum(results["hat_values"])
    assert np.isclose(leverage_sum, len(beta_hat), atol=1e-8)


def test_verification_against_sklearn():
    if LinearRegression is None:
        return

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 3))
    beta_true = np.array([2.0, -1.2, 0.8, 3.5])
    X_ = np.hstack([np.ones((60, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.5, size=60)

    beta_hat = np.linalg.solve(X_.T @ X_, X_.T @ y)
    ours = residual_plots(X, y, beta_hat)
    plt.close("all")

    skl = _verification_with_sklearn(X, y)

    # Vì sklearn.fit_intercept=True còn core code dùng ma trận design đã thêm intercept,
    # các fitted values và residuals phải gần bằng nhau.
    assert np.allclose(ours["y_hat"], skl["y_hat"], atol=1e-8)
    assert np.allclose(ours["residuals"], skl["residuals"], atol=1e-8)


if __name__ == "__main__":
    print("Đang chạy kiểm thử cho file residual_analysis.py...")
    test_residual_plots_no_error()
    print(" -> test_residual_plots_no_error: PASSED")
    test_residual_mean_near_zero()
    print(" -> test_residual_mean_near_zero: PASSED")
    test_residual_plots_with_outlier()
    print(" -> test_residual_plots_with_outlier: PASSED")
    test_leverage_property()
    print(" -> test_leverage_property: PASSED")
    test_leverage_trace_property()
    print(" -> test_leverage_trace_property: PASSED")
    test_verification_against_sklearn()
    print(" -> test_verification_against_sklearn: PASSED")
    print("\nAll tests passed!")

    rng_demo = np.random.default_rng(42)
    X_demo = rng_demo.normal(size=(100, 3))
    beta_demo = np.array([2.0, -1.2, 0.8, 3.5])
    X_demo_ = np.hstack([np.ones((100, 1)), X_demo])
    y_demo = X_demo_ @ beta_demo + rng_demo.normal(scale=0.5, size=100)
    beta_hat_demo = np.linalg.solve(X_demo_.T @ X_demo_, X_demo_.T @ y_demo)
    print("\n[Demo] Hiển thị 4 biểu đồ phân tích phần dư...")
    residual_plots(X_demo, y_demo, beta_hat_demo)