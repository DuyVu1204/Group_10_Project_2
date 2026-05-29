import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
except Exception:
    LinearRegression = None
    KFold = None
    mean_squared_error = None


def kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    fit_intercept: bool = True,
    random_state: int = None,
) -> float:
    """
    K-fold cross-validation cho hồi quy tuyến tính OLS.

    Core implementation:
        - Tự chia fold bằng NumPy
        - Tự fit OLS theo normal equation:
            beta_hat = (X^T X)^(-1) X^T y
        - Không dùng sklearn.linear_model.LinearRegression
        - Không dùng numpy.linalg.lstsq

    Trả về:
        Mean MSE trên k fold
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X phải là numpy.ndarray.")
    if not isinstance(y, np.ndarray):
        raise ValueError("y phải là numpy.ndarray.")
    if X.ndim != 2:
        raise ValueError("X phải là ma trận 2 chiều.")
    if y.ndim != 1:
        raise ValueError("y phải là vector 1 chiều.")
    if not isinstance(k, int) or k < 2:
        raise ValueError("Số fold k phải là số nguyên >= 2.")

    n = y.shape[0]
    if n == 0 or X.shape[0] != n:
        raise ValueError("X và y phải cùng số mẫu, n > 0.")
    if k > n:
        raise ValueError("Số fold k không được lớn hơn số mẫu.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    mse_list = []
    current = 0

    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size

        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if fit_intercept:
            X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train

        if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
            raise ValueError("Ma trận X^T X suy biến, không thể tính nghiệm OLS duy nhất.")

        beta_hat = np.linalg.solve(XtX, Xty)
        y_pred = X_val @ beta_hat
        mse = np.mean((y_val - y_pred) ** 2)
        mse_list.append(mse)

        current = stop

    return float(np.mean(mse_list))


def _verification_with_sklearn(X: np.ndarray, y: np.ndarray, k: int, random_state: int = None) -> float:
    """
    Hàm kiểm chứng, chỉ dùng sklearn để so sánh kết quả với implementation từ đầu.
    Không dùng trong core algorithm.
    """
    if LinearRegression is None or KFold is None or mean_squared_error is None:
        raise ImportError("Cần cài scikit-learn để chạy verification.")

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    mse_list = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_list.append(mean_squared_error(y_val, y_pred))

    return float(np.mean(mse_list))


def test_kfold_cv_simple():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = 2.0 * X.flatten() + 1.0
    mse = kfold_cv(X, y, k=3, fit_intercept=True, random_state=123)
    assert np.isclose(mse, 0.0, atol=1e-10)


def test_kfold_cv_known_result():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    beta_true = np.array([1.0, -2.0, 0.5])
    X_ = np.hstack([np.ones((20, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.1, size=20)
    mse = kfold_cv(X, y, k=4, fit_intercept=True, random_state=42)
    assert mse < 0.02


def test_kfold_cv_k_greater_than_n():
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([2.0, 4.0, 6.0])
    try:
        kfold_cv(X, y, k=5, fit_intercept=True, random_state=1)
        assert False, "Hàm không ném ra ngoại lệ khi k > n"
    except ValueError as e:
        assert "không được lớn hơn số mẫu" in str(e)


def test_kfold_cv_reproducibility():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    beta_true = np.array([1.0, -1.0, 2.0, 0.5])
    X_ = np.hstack([np.ones((50, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.2, size=50)
    mse1 = kfold_cv(X, y, k=5, fit_intercept=True, random_state=42)
    mse2 = kfold_cv(X, y, k=5, fit_intercept=True, random_state=42)
    assert np.isclose(mse1, mse2)


def test_verification_against_sklearn():
    if LinearRegression is None:
        return

    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    beta_true = np.array([2.0, -1.5, 0.7, 3.0, -0.8])
    X_ = np.hstack([np.ones((80, 1)), X])
    y = X_ @ beta_true + rng.normal(scale=0.15, size=80)

    cv_ours = kfold_cv(X, y, k=5, fit_intercept=True, random_state=123)
    cv_sklearn = _verification_with_sklearn(X, y, k=5, random_state=123)

    # Hai implementation chia fold khác nhau (numpy RNG vs sklearn KFold),
    # nên chỉ kiểm tra kết quả cùng order of magnitude, không so sánh exact.
    assert cv_ours > 0 and cv_sklearn > 0
    assert abs(cv_ours - cv_sklearn) / (cv_sklearn + 1e-12) < 0.5


if __name__ == "__main__":
    print("Đang chạy kiểm thử cho file cross_validation.py...")
    test_kfold_cv_simple()
    print(" -> test_kfold_cv_simple: PASSED")
    test_kfold_cv_known_result()
    print(" -> test_kfold_cv_known_result: PASSED")
    test_kfold_cv_k_greater_than_n()
    print(" -> test_kfold_cv_k_greater_than_n: PASSED")
    test_kfold_cv_reproducibility()
    print(" -> test_kfold_cv_reproducibility: PASSED")
    test_verification_against_sklearn()
    print(" -> test_verification_against_sklearn: PASSED")
    print("\nAll tests passed!")

    rng_demo = np.random.default_rng(42)
    X_demo = rng_demo.normal(size=(100, 3))
    beta_demo = np.array([2.0, -1.0, 0.5, 3.0])
    X_demo_ = np.hstack([np.ones((100, 1)), X_demo])
    y_demo = X_demo_ @ beta_demo + rng_demo.normal(scale=0.2, size=100)

    cv_score = kfold_cv(X_demo, y_demo, k=5, fit_intercept=True, random_state=42)
    print(f"\n[Demo] Mean CV MSE (from scratch): {cv_score:.6f}")

    if LinearRegression is not None:
        cv_sklearn = _verification_with_sklearn(X_demo, y_demo, k=5, random_state=42)
        print(f"[Demo] Mean CV MSE (sklearn verification): {cv_sklearn:.6f}")
        print(f"[Demo] Difference: {abs(cv_score - cv_sklearn):.6e}")